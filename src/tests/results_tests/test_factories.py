from sqlalchemy import text

from .factories import (
    ModelGroupFactory,
    ModelFactory,
    SubsetFactory,
    EvaluationFactory,
    PredictionFactory,
    IndividualImportanceFactory,
)


def test_evaluation_factories_no_subset(db_engine_with_results_schema, db_session):
    #Base.metadata.create_all(db_engine_with_results_schema)

    model_group = ModelGroupFactory()
    model = ModelFactory(model_group_rel=model_group)
    for metric, value in [("precision@", 0.4), ("recall@", 0.3)]:
        EvaluationFactory(
            model_rel=model, 
            metric=metric, 
            parameter="100_abs", 
            stochastic_value=value,
        )
    db_session.commit()
    
    with db_engine_with_results_schema.connect() as conn:
        results = conn.execute(
            text(
                """
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
        )

        rows = list(results)
        assert len(rows) == 2  # Should have 2 evaluations
        
        for row in rows:
            model_group_id = row.model_group_id
            model_id = row.model_id
            metric = row.metric
            value = row.stochastic_value
            subset_hash = row.subset_hash
            
            # if the evaluations are created with the model group and model,
            # as opposed to an autoprovisioned one,
            # the ids in a fresh DB should be 1
            assert model_group_id == 1
            assert model_id == 1
            assert not subset_hash

    
    # for model_group_id, model_id, metric, value, subset_hash in results:
    #     # if the evaluations are created with the model group and model,
    #     # as opposed to an autoprovisioned one,
    #     # the ids in a fresh DB should be 1
    #     assert model_group_id == 1
    #     assert model_id == 1
    #     assert not subset_hash


def test_evaluation_factories_with_subset(db_engine_with_results_schema, db_session):
    #Base.metadata.create_all(db_engine_with_results_schema)

    model_group = ModelGroupFactory()
    model = ModelFactory(model_group_rel=model_group)
    subset = SubsetFactory()
    for metric, value in [("precision@", 0.4), ("recall@", 0.3)]:
        EvaluationFactory(
            model_rel=model, 
            metric=metric, 
            parameter="100_abs", 
            stochastic_value=value,
            subset_hash=subset.subset_hash,
        )
    db_session.commit()
    
    with db_engine_with_results_schema.connect() as conn:
        results = conn.execute(
            text(
            """
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
        )

        rows = list(results)
        print(len(rows))
        assert len(rows) == 2

    for row in rows:
        # if the evaluations are created with the model group and model,
        # as opposed to an autoprovisioned one,
        # the ids in a fresh DB should be 1
        assert row.model_group_id == 1
        assert row.model_id == 1
        assert row.s_subset_hash == row.e_subset_hash


def test_prediction_factories(db_engine_with_results_schema, db_session):
    
        #Base.metadata.create_all(engine)
        
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
                model_rel=model, 
                entity_id=entity_id, 
                as_of_date=as_of_date,
            )

        # create some basic predictions, but with the same model group and
        # model to test the factory relationships
        for entity_id, as_of_date in entity_dates:
            PredictionFactory(
                model_rel=model, entity_id=entity_id, as_of_date=as_of_date
            )
        db_session.commit()

        with db_engine_with_results_schema.connect() as conn:
            results = conn.execute(
                text("""
                    select m.*, p.*
                    from
                        test_results.predictions p
                        join triage_metadata.models m using (model_id)
                        join test_results.individual_importances i using (model_id, entity_id, as_of_date)
                    """
                )
            )
            assert len([row for row in results]) == 6
            # if the predictions are created with the model,
            # the join should work and we should have the original six results
