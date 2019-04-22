import sqlalchemy.sql.expression as ex
from sqlalchemy.ext.compiler import compiles


def make_sql_clause(s, constructor):
    if not isinstance(s, ex.ClauseElement):
        return constructor(s)
    else:
        return s


class CreateTableAs(ex.Executable, ex.ClauseElement):
    def __init__(self, name, query):
        self.name = name
        self.query = query


@compiles(CreateTableAs)
def _create_table_as(element, compiler, **kw):
    return "CREATE TABLE %s AS %s" % (element.name, compiler.process(element.query))


class InsertFromSelect(ex.Executable, ex.ClauseElement):
    def __init__(self, name, query):
        self.name = name
        self.query = query


@compiles(InsertFromSelect)
def _insert_from_select(element, compiler, **kw):
    return "INSERT INTO %s (%s)" % (element.name, compiler.process(element.query))


def to_sql_name(name):
    return name.replace('"', "")
