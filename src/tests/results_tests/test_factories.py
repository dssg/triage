import testing.postgresql
from sqlalchemy import create_engine

from triage.component.results_schema import Base

from .factories import (
    ModelGroupFactory,
    ModelFactory,
    EvaluationFactory,
    PredictionFactory,
    IndividualImportanceFactory,
    init_engine,
    session,
)


def test_evaluation_factories():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        Base.metadata.create_all(engine)
        init_engine(engine)

        model_group = ModelGroupFactory()
        model = ModelFactory(model_group_rel=model_group)
        for metric, value in [("precision@", 0.4), ("recall@", 0.3)]:
            EvaluationFactory(
                model_rel=model, metric=metric, parameter="100_abs", value=value
            )
        session.commit()
        results = engine.execute(
            """
            select
                model_group_id,
                m.model_id,
                e.metric,
                e.value
            from
                test_results.evaluations e
                join model_metadata.models m using (model_id)
                join model_metadata.matrices test_matrix on (e.matrix_uuid = test_matrix.matrix_uuid)
        """
        )
        for model_group_id, model_id, metric, value in results:
            # if the evaluations are created with the model group and model,
            # as opposed to an autoprovisioned one,
            # the ids in a fresh DB should be 1
            assert model_group_id == 1
            assert model_id == 1


def test_prediction_factories():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        Base.metadata.create_all(engine)
        init_engine(engine)

        # create some basic predictions, but with the same model group and
        # model to test the factory relationships
        model_group = ModelGroupFactory()
        model = ModelFactory(model_group_rel=model_group)
        for entity_id, as_of_date in [
            (1, "2016-01-01"),
            (1, "2016-04-01"),
            (2, "2016-01-01"),
            (2, "2016-04-01"),
            (3, "2016-01-01"),
            (3, "2016-04-01"),
        ]:
            PredictionFactory(
                model_rel=model, entity_id=entity_id, as_of_date=as_of_date
            )
            IndividualImportanceFactory(
                model_rel=model, entity_id=entity_id, as_of_date=as_of_date
            )
        session.commit()
        results = engine.execute(
            """
            select m.*, p.*
            from
                test_results.predictions p
                join model_metadata.models m using (model_id)
                join test_results.individual_importances i using (model_id, entity_id, as_of_date)
        """
        )
        assert len([row for row in results]) == 6
        # if the predictions are created with the model,
        # the join should work and we should have the original six results
