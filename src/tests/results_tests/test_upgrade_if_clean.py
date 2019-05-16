from triage.component import results_schema
from alembic import command, script
import pytest


def test_upgrade_if_clean_upgrades_if_clean(db_engine):
    results_schema.upgrade_if_clean(db_engine.url)
    db_version = db_engine.execute("select version_num from results_schema_versions").scalar()
    alembic_cfg = results_schema.alembic_config(db_engine.url)
    assert db_version == script.ScriptDirectory.from_config(alembic_cfg).get_current_head()


def test_upgrade_if_clean_does_not_upgrade_if_not_clean(db_engine):
    command.upgrade(results_schema.alembic_config(dburl=db_engine.url), "head")
    command.downgrade(results_schema.alembic_config(dburl=db_engine.url), "-1")
    with pytest.raises(ValueError):
        results_schema.upgrade_if_clean(db_engine.url)
