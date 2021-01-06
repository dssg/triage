import testing.postgresql
from sqlalchemy import create_engine

from triage.component.results_schema import Base

from .factories import (
    ModelGroupFactory,
    ModelFactory,
    SubsetFactory,
    EvaluationFactory,
    PredictionFactory,
    IndividualImportanceFactory,
    init_engine,
    session,
)


def test_evaluation_factories_no_subset():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        Base.metadata.create_all(engine)
        init_engine(engine)

        model_group = ModelGroupFactory()
        model = ModelFactory(model_group_rel=model_group)
        for metric, value in [("precision@", 0.4), ("recall@", 0.3)]:
            EvaluationFactory(
                model_rel=model, metric=metric, parameter="100_abs", stochastic_value=value
            )
        session.commit()
        results = engine.execute(
            """\
            select
                model_group_id,
                m.model_id,
                e.metric,
                e.stochastic_value,
                e.subset_hash
            from
                test_results.evaluations e
                join triage_metadata.models m using (model_id)
            """
        )
        for model_group_id, model_id, metric, value, subset_hash in results:
            # if the evaluations are created with the model group and model,
            # as opposed to an autoprovisioned one,
            # the ids in a fresh DB should be 1
            assert model_group_id == 1
            assert model_id == 1
            assert not subset_hash


def test_evaluation_factories_with_subset(db_engine):
    Base.metadata.create_all(db_engine)
    init_engine(db_engine)

    model_group = ModelGroupFactory()
    model = ModelFactory(model_group_rel=model_group)
    subset = SubsetFactory()
    for metric, value in [("precision@", 0.4), ("recall@", 0.3)]:
        EvaluationFactory(
            model_rel=model, metric=metric, parameter="100_abs", stochastic_value=value
        )
    session.commit()
    results = db_engine.execute(
        """\
        select
            model_group_id,
            m.model_id,
            s.subset_hash as s_subset_hash,
            e.subset_hash as e_subset_hash,
            e.metric,
            e.stochastic_value
        from
            test_results.evaluations e
            join triage_metadata.models m using (model_id)
            join triage_metadata.subsets s using (subset_hash)
        """
    )
    for model_group_id, model_id, s_subset_hash, e_subset_hash, metric, value in results:
        # if the evaluations are created with the model group and model,
        # as opposed to an autoprovisioned one,
        # the ids in a fresh DB should be 1
        assert model_group_id == 1
        assert model_id == 1
        assert s_subset_hash == e_subset_hash


def test_prediction_factories():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        Base.metadata.create_all(engine)
        init_engine(engine)
        
        model_group = ModelGroupFactory()
        model = ModelFactory(model_group_rel=model_group)
       
        entity_dates = [
            (1, "2016-01-01"),
            (1, "2016-04-01"),
            (2, "2016-01-01"),
            (2, "2016-04-01"),
            (3, "2016-01-01"),
            (3, "2016-04-01"),
        ]

        for entity_id, as_of_date in entity_dates:
            IndividualImportanceFactory(
                model_rel=model, entity_id=entity_id, as_of_date=as_of_date
            )
        session.commit()

        # create some basic predictions, but with the same model group and
        # model to test the factory relationships
        for entity_id, as_of_date in entity_dates:
            PredictionFactory(
                model_rel=model, entity_id=entity_id, as_of_date=as_of_date
            )
        session.commit()

        results = engine.execute(
            f"""
            select m.*, p.*
            from
                test_results.predictions p
                join triage_metadata.models m using (model_id)
                join test_results.individual_importances i using (model_id, entity_id, as_of_date)
            """
        )
        assert len([row for row in results]) == 6
        # if the predictions are created with the model,
        # the join should work and we should have the original six results
