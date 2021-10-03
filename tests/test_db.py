import unittest
import sqlite3

from db.galaxy import Galaxy, GalaxiesDB, DBError


class DBTests(unittest.TestCase):
    def test_no_unexpected_tables(self):
        conn = sqlite3.connect(':memory:')
        conn.execute('CREATE TABLE unexpected(TEXT);')
        self.assertRaisesRegex(DBError, r'.+unexpected.*', lambda : GalaxiesDB(conn))

    def test_creates_table(self):
        db = GalaxiesDB(':memory:')
        cursor = db.con.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        names = cursor.fetchall()
        self.assertListEqual(names, [('galaxies', )])

    def test_no_table_recreate(self):
        conn = sqlite3.connect(':memory:')
        db = GalaxiesDB(conn)
        db.save(Galaxy(1, 1, 1, 1, 1))
        db = GalaxiesDB(conn)
        cursor = db.con.execute('SELECT * FROM galaxies')
        self.assertEqual(1, len(cursor.fetchall()))
        db.close()

    def test_add_many(self):
        db = GalaxiesDB(':memory:')
        n = 10
        db.save([Galaxy(1, 1, 1, 1, 1) for i in range(n)])
        cursor = db.con.execute('SELECT * FROM galaxies')
        self.assertEqual(n, len(cursor.fetchall()))
        db.close()

    def test_query_db(self):
        db = GalaxiesDB(':memory:')
        n = 4
        db.save([Galaxy(i, 1, 1, 1, i % 2) for i in range(n)])
        _all = db.find(Galaxy)
        self.assertEqual(n, len(_all))
        _even = db.find(Galaxy, 'dist_error = 0.0')
        self.assertTrue(all(map(lambda g: g.ed == 0, _even)))
        db.close()

    def test_drop_db(self):
        db = GalaxiesDB(':memory:')
        n = 10
        db.save([Galaxy(1, 1, 1, 1, 1) for i in range(n)])
        db.drop(Galaxy)
        self.assertEqual(0, len(db.find(Galaxy)))
        db.close()
