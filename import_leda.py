import argparse
import re
import sqlite3
from numpy import pi

from db._db_abc import TableDesr
from db.galaxy import Galaxy, GalaxiesDB


def decode_file(filename: str):
    types = (
        {'type': str, 'required': False},
        {'type': str, 'required': True},
        {'type': str, 'required': True},
        {'type': float, 'required': True},
        {'type': float, 'required': True},
        {'type': float, 'required': False},
        {'type': float, 'required': False},
        {'type': str, 'required': False},
        {'type': float, 'required': True},
        {'type': float, 'required': False},
        {'type': float, 'required': False},
        {'type': str, 'required': False},
        {'type': float, 'required': False},
    )
    db = sqlite3.connect(':memory:')
    with open(filename) as f:
        # read header for colum names and sizes
        header = f.readline()
        columns = header.split('|')
        # col_lens = list(map(lambda el: len(el), columns))
        # col_lens[-1] -= 1
        columns = list(map(lambda el: el.strip(), columns))
        typespec = ''
        for col, spec, name in zip(columns, types, columns):
            descr = '\t%s %s' % (name, TableDesr.Field.py_to_sqlite_types()[spec['type']])
            if spec['required']:
                descr += ' NOT NULL'
            typespec += '%s,\n' % descr
        typespec = typespec[:-2]
        db.cursor().execute('CREATE TABLE raw(%s);' % typespec)

        # skip divider
        line = f.readline()
        if re.match(r'^[-+]+$', line) is not None:
            line = f.readline()

        # read the data in sqlite3 database
        while re.match(r'\([0-9]+ rows\)', line) is None:
            raw_values = line.split('|')
            raw_values = list(map(lambda el: el.strip(), raw_values))
            db.cursor().execute('INSERT INTO raw VALUES(%s);' % (len(columns) * '?,')[:-1], raw_values)
            line = f.readline()

    return db


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Import galaxies data from hyperleda database.')
    parser.add_argument('-f', '--file', type=str, required=True)
    parser.add_argument('-a', '--append', action='store_true', default=False)
    parser.add_argument('-o', '--out', type=str, default='out.db')
    args = parser.parse_args()

    db = decode_file(args.file)
    curs = db.execute(
        'SELECT name, Dist, Ra, dec, error, modulus, LK '
        'FROM raw'
        ' WHERE method IN ("TRGB", "Cep", "HB", "RR", "geom")'
    )
    galaxies = []
    for data in curs:
        name, dist, ra, dec, err, mod, lk = data
        ra = ra * 15 / 180 * pi
        dec = dec / 180 * pi
        mass = lk
        ed = 0
        galaxy = Galaxy(dist, ra, dec, mass, ed, _id=name)
        galaxies.append(galaxy)

    db = GalaxiesDB(args.out)
    if not args.append:
        db.drop(Galaxy)
    db.save(galaxies)
    db.close()
