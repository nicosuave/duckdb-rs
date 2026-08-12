use std::{
    ffi::{CStr, CString},
    mem,
    os::raw::c_char,
    ptr, str,
    sync::{Arc, Mutex},
};

use super::{Appender, Config, Connection, DmlStatementType, QueryAppender, Result, ffi};
use crate::{
    core::LogicalTypeHandle,
    error::{
        Error, duckdb_failure_from_message, result_from_duckdb_appender, result_from_duckdb_extract,
        result_from_duckdb_prepare, result_from_duckdb_result,
    },
    raw_statement::RawStatement,
    statement::Statement,
};

#[derive(Debug)]
struct DatabaseHandle {
    db: ffi::duckdb_database,
    close_on_drop: bool,
}

// `duckdb_database` is an opaque C pointer. We share it via `Arc` so the database
// is closed exactly once when the last connection referencing it is dropped.
// This means the handle may be dropped on any thread that owns a `Connection`.
unsafe impl Send for DatabaseHandle {}

impl DatabaseHandle {
    #[inline]
    pub fn new(db: ffi::duckdb_database, close_on_drop: bool) -> Self {
        Self { db, close_on_drop }
    }

    #[inline]
    pub fn raw(&self) -> ffi::duckdb_database {
        self.db
    }
}

impl Drop for DatabaseHandle {
    fn drop(&mut self) {
        if !self.close_on_drop {
            return;
        }
        if self.db.is_null() {
            return;
        }
        unsafe {
            ffi::duckdb_close(&mut self.db);
            self.db = ptr::null_mut();
        }
    }
}

pub struct InnerConnection {
    // Mutex makes the Send-only handle `Sync` so the `Arc` can be shared across threads;
    // it is not used for concurrent access to the database itself.
    database: Arc<Mutex<DatabaseHandle>>,
    pub con: ffi::duckdb_connection,
    interrupt: Arc<InterruptHandle>,
}

impl InnerConnection {
    #[inline]
    unsafe fn new(database: Arc<Mutex<DatabaseHandle>>) -> Result<Self> {
        unsafe {
            let mut con: ffi::duckdb_connection = ptr::null_mut();
            let db_raw = database.lock().expect("database handle mutex poisoned").raw();
            let r = ffi::duckdb_connect(db_raw, &mut con);
            if r != ffi::DuckDBSuccess {
                ffi::duckdb_disconnect(&mut con);
                return Err(Error::DuckDBFailure(
                    ffi::Error::new(r),
                    Some("connect error".to_owned()),
                ));
            }
            let interrupt = Arc::new(InterruptHandle::new(con));

            Ok(Self {
                database,
                con,
                interrupt,
            })
        }
    }

    pub fn open_with_flags(c_path: &CStr, config: Config) -> Result<Self> {
        unsafe {
            let mut db: ffi::duckdb_database = ptr::null_mut();
            let mut c_err = std::ptr::null_mut();
            let r = ffi::duckdb_open_ext(c_path.as_ptr(), &mut db, config.duckdb_config(), &mut c_err);
            if r != ffi::DuckDBSuccess {
                let msg = ffi::DuckDbString::from_nullable_ptr(c_err).map(|c_err| c_err.to_string_lossy().to_string());
                return Err(Error::DuckDBFailure(ffi::Error::new(r), msg));
            }
            Self::new_from_raw_db(db, true)
        }
    }

    #[inline]
    pub(crate) unsafe fn new_from_raw_db(raw: ffi::duckdb_database, close_on_drop: bool) -> Result<Self> {
        unsafe { Self::new(Arc::new(Mutex::new(DatabaseHandle::new(raw, close_on_drop)))) }
    }

    pub fn close(&mut self) -> Result<()> {
        if self.con.is_null() {
            return Ok(());
        }
        unsafe {
            ffi::duckdb_disconnect(&mut self.con);
            self.con = ptr::null_mut();
            self.interrupt.clear();
        }
        Ok(())
    }

    /// Creates a new connection to the already-opened database.
    pub fn try_clone(&self) -> Result<Self> {
        unsafe { Self::new(self.database.clone()) }
    }

    pub fn execute(&mut self, sql: &str) -> Result<()> {
        let c_str = CString::new(sql)?;
        unsafe {
            let mut out = mem::zeroed();
            let r = ffi::duckdb_query(self.con, c_str.as_ptr() as *const c_char, &mut out);
            result_from_duckdb_result(r, &mut out)?;
            ffi::duckdb_destroy_result(&mut out);
            Ok(())
        }
    }

    pub fn prepare<'a>(&mut self, conn: &'a Connection, sql: &str) -> Result<Statement<'a>> {
        let c_str = CString::new(sql)?;

        // Extract statements (handles both single and multi-statement queries)
        let mut extracted = ptr::null_mut();
        let num_stmts =
            unsafe { ffi::duckdb_extract_statements(self.con, c_str.as_ptr() as *const c_char, &mut extracted) };
        result_from_duckdb_extract(num_stmts, extracted)?;

        // Auto-cleanup on drop
        let _guard = ExtractedStatementsGuard(extracted);

        // Execute all intermediate statements
        for i in 0..num_stmts - 1 {
            self.execute_extracted_statement(extracted, i)?;
        }

        // Prepare and return final statement
        let final_stmt = self.prepare_extracted_statement(extracted, num_stmts - 1)?;
        Ok(Statement::new(conn, unsafe { RawStatement::new(final_stmt) }))
    }

    pub fn validate_single_dml_statement(&mut self, sql: &str) -> Result<DmlStatementType> {
        let c_str = CString::new(sql)?;
        let mut extracted = ptr::null_mut();
        let num_stmts = unsafe { ffi::duckdb_extract_statements(self.con, c_str.as_ptr(), &mut extracted) };
        result_from_duckdb_extract(num_stmts, extracted)?;
        let _guard = ExtractedStatementsGuard(extracted);
        if num_stmts != 1 {
            return Err(duckdb_failure_from_message(format!(
                "expected exactly one statement, found {num_stmts}"
            )));
        }

        // Preparing the extracted statement binds and classifies it, but does
        // not execute it. In particular, no intermediate statement is run.
        let mut statement = self.prepare_extracted_statement(extracted, 0)?;
        let statement_type = unsafe { ffi::duckdb_prepared_statement_type(statement) };
        unsafe { ffi::duckdb_destroy_prepare(&mut statement) };
        match statement_type {
            ffi::duckdb_statement_type_DUCKDB_STATEMENT_TYPE_INSERT => Ok(DmlStatementType::Insert),
            ffi::duckdb_statement_type_DUCKDB_STATEMENT_TYPE_UPDATE => Ok(DmlStatementType::Update),
            ffi::duckdb_statement_type_DUCKDB_STATEMENT_TYPE_DELETE => Ok(DmlStatementType::Delete),
            ffi::duckdb_statement_type_DUCKDB_STATEMENT_TYPE_MERGE_INTO => Ok(DmlStatementType::MergeInto),
            _ => Err(duckdb_failure_from_message(format!(
                "statement type {statement_type} is not INSERT, UPDATE, DELETE, or MERGE INTO"
            ))),
        }
    }

    fn prepare_extracted_statement(
        &self,
        extracted: ffi::duckdb_extracted_statements,
        index: ffi::idx_t,
    ) -> Result<ffi::duckdb_prepared_statement> {
        let mut stmt = ptr::null_mut();
        let res = unsafe { ffi::duckdb_prepare_extracted_statement(self.con, extracted, index, &mut stmt) };
        result_from_duckdb_prepare(res, stmt)?;
        Ok(stmt)
    }

    fn execute_extracted_statement(
        &self,
        extracted: ffi::duckdb_extracted_statements,
        index: ffi::idx_t,
    ) -> Result<()> {
        let mut stmt = self.prepare_extracted_statement(extracted, index)?;

        let mut result = unsafe { mem::zeroed() };
        let rc = unsafe { ffi::duckdb_execute_prepared(stmt, &mut result) };

        let error = if rc != ffi::DuckDBSuccess {
            unsafe {
                let c_err = ffi::duckdb_result_error(&mut result as *mut _);
                let msg = if c_err.is_null() {
                    None
                } else {
                    Some(CStr::from_ptr(c_err).to_string_lossy().to_string())
                };
                Some(Error::DuckDBFailure(ffi::Error::new(rc), msg))
            }
        } else {
            None
        };

        unsafe {
            ffi::duckdb_destroy_prepare(&mut stmt);
            ffi::duckdb_destroy_result(&mut result);
        }

        error.map_or(Ok(()), Err)
    }

    pub fn appender<'a>(
        &mut self,
        conn: &'a Connection,
        catalog: Option<&str>,
        schema: &str,
        table: &str,
    ) -> Result<Appender<'a>> {
        let c_app = self.create_table_appender(catalog, schema, table)?;
        Ok(Appender::new(conn, c_app))
    }

    fn create_table_appender(
        &mut self,
        catalog: Option<&str>,
        schema: &str,
        table: &str,
    ) -> Result<ffi::duckdb_appender> {
        let mut c_app: ffi::duckdb_appender = ptr::null_mut();
        let c_catalog = catalog.map(CString::new).transpose()?;
        let c_table = CString::new(table)?;
        let c_schema = CString::new(schema)?;
        let r = unsafe {
            ffi::duckdb_appender_create_ext(
                self.con,
                c_catalog.as_ref().map_or(ptr::null(), |c| c.as_ptr() as *const c_char),
                c_schema.as_ptr() as *const c_char,
                c_table.as_ptr() as *const c_char,
                &mut c_app,
            )
        };
        result_from_duckdb_appender(r, &mut c_app)?;
        Ok(c_app)
    }

    pub fn appender_with_columns<'a>(
        &mut self,
        conn: &'a Connection,
        catalog: Option<&str>,
        schema: &str,
        table: &str,
        columns: &[&str],
    ) -> Result<Appender<'a>> {
        // The C API only supports narrowing columns after the appender is created.
        // Create the appender first, then activate the requested column subset.
        let c_app = self.create_table_appender(catalog, schema, table)?;
        let mut appender = Appender::new(conn, c_app);
        for column in columns {
            appender.add_column(column)?;
        }
        Ok(appender)
    }

    pub fn query_appender<'a>(
        &mut self,
        conn: &'a Connection,
        query: &str,
        types: &[LogicalTypeHandle],
        relation_name: &str,
        column_names: &[&str],
    ) -> Result<QueryAppender<'a>> {
        let runtime_version = unsafe { CStr::from_ptr(ffi::duckdb_library_version()) }.to_string_lossy();
        if runtime_version != "v1.5.5" {
            return Err(duckdb_failure_from_message(format!(
                "query appender compatibility layer requires DuckDB v1.5.5, found {runtime_version}"
            )));
        }
        if types.len() != column_names.len() {
            return Err(duckdb_failure_from_message(format!(
                "query appender schema has {} types but {} column names",
                types.len(),
                column_names.len()
            )));
        }

        let c_extract_query = CString::new(query)?;
        let mut extracted = ptr::null_mut();
        let num_stmts = unsafe { ffi::duckdb_extract_statements(self.con, c_extract_query.as_ptr(), &mut extracted) };
        result_from_duckdb_extract(num_stmts, extracted)?;
        let _guard = ExtractedStatementsGuard(extracted);
        if num_stmts != 1 {
            return Err(duckdb_failure_from_message(format!(
                "query appender requires exactly one statement, found {num_stmts}"
            )));
        }

        let c_query = c_extract_query;
        let c_relation_name = CString::new(relation_name)?;
        let c_column_names = column_names
            .iter()
            .map(|name| CString::new(*name))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut column_name_ptrs = c_column_names.iter().map(|name| name.as_ptr()).collect::<Vec<_>>();
        let mut type_ptrs = types.iter().map(|logical_type| logical_type.ptr).collect::<Vec<_>>();
        let mut c_app = ptr::null_mut();
        let rc = unsafe {
            ffi::duckdb_appender_create_query(
                self.con,
                c_query.as_ptr(),
                type_ptrs.len() as ffi::idx_t,
                type_ptrs.as_mut_ptr(),
                c_relation_name.as_ptr(),
                column_name_ptrs.as_mut_ptr(),
                &mut c_app,
            )
        };
        result_from_duckdb_appender(rc, &mut c_app)?;
        Ok(QueryAppender::new(conn, c_app))
    }

    pub fn get_interrupt_handle(&self) -> Arc<InterruptHandle> {
        self.interrupt.clone()
    }

    #[inline]
    pub fn is_autocommit(&self) -> bool {
        true
    }
}

struct ExtractedStatementsGuard(ffi::duckdb_extracted_statements);

impl Drop for ExtractedStatementsGuard {
    fn drop(&mut self) {
        unsafe { ffi::duckdb_destroy_extracted(&mut self.0) }
    }
}

impl Drop for InnerConnection {
    #[inline]
    fn drop(&mut self) {
        use std::thread::panicking;
        if let Err(e) = self.close() {
            if panicking() {
                eprintln!("Error while closing DuckDB connection: {e:?}");
            } else {
                panic!("Error while closing DuckDB connection: {e:?}");
            }
        }
    }
}

/// A handle that allows interrupting long-running queries.
pub struct InterruptHandle {
    conn: Mutex<ffi::duckdb_connection>,
}

unsafe impl Send for InterruptHandle {}
unsafe impl Sync for InterruptHandle {}

impl InterruptHandle {
    fn new(conn: ffi::duckdb_connection) -> Self {
        Self { conn: Mutex::new(conn) }
    }

    fn clear(&self) {
        *(self.conn.lock().unwrap()) = ptr::null_mut();
    }

    /// Interrupt the query currently running on the connection this handle was
    /// obtained from. The interrupt will cause that query to fail with
    /// `Error::DuckDBFailure`. If the connection was dropped after obtaining
    /// this interrupt handle, calling this method results in a noop.
    ///
    /// See [`crate::Connection::interrupt_handle`] for an example.
    pub fn interrupt(&self) {
        let db_handle = self.conn.lock().unwrap();

        if !db_handle.is_null() {
            unsafe {
                ffi::duckdb_interrupt(*db_handle);
            }
        }
    }
}
