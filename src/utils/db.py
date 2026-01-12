import duckdb

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None

    def connect(self):
        if not self.conn:
            self.conn = duckdb.connect(self.db_path)
            self.conn.execute("SET memory_limit = '4GB'")
            self.conn.execute("INSTALL spatial; LOAD spatial;")
        return self.conn

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None