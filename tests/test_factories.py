import testing.postgresql
from sqlalchemy import create_engine
from results_schema import schema
from results_schema.factories import ModelGroupFactory,\
    ModelFactory,\
    EvaluationFactory,\
    init_engine,\
    session


def test_factories():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        schema.Base.metadata.create_all(engine)
        init_engine(engine)

        # create some basic evaluations, but with the same model group and
        # model to test the factory relationships
        model_group = ModelGroupFactory()
        model = ModelFactory(model_group_rel=model_group)
        for metric, value in [
            ('precision@', 0.4),
            ('recall@', 0.3),
        ]:
            EvaluationFactory(
                model_rel=model,
                metric=metric,
                parameter='100_abs',
                value=value
            )
        session.commit()
        results = engine.execute('''
            select
                model_group_id,
                m.model_id,
                e.metric,
                e.value
            from
                results.evaluations e
                join results.models m using (model_id)
        ''')
        for model_group_id, model_id, metric, value in results:
            # if the evaluations are created with the model group and model,
            # as opposed to an autoprovisioned one,
            # the ids in a fresh DB should be 1
            assert model_group_id == 1
            assert model_id == 1
