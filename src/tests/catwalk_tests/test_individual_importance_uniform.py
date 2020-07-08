import pandas as pd

from triage.component.catwalk.individual_importance.uniform import uniform_distribution
from tests.utils import rig_engines, get_matrix_store, matrix_metadata_creator
import datetime

from tests.results_tests.factories import (
    ModelFactory,
    FeatureImportanceFactory,
)


def test_uniform_distribution():
    with rig_engines() as (db_engine, project_storage):
        model = ModelFactory()
        feature_importances = [
            FeatureImportanceFactory(model_rel=model, feature="feature_{}".format(i))
            for i in range(0, 10)
        ]
        data_dict = {"entity_id": [1, 1], "as_of_date": ["2016-01-01", "2017-01-01"], "label": [0, 1]}
        for imp in feature_importances:
            data_dict[imp.feature] = [0.5, 0.5]
        metadata = matrix_metadata_creator()
        test_store = get_matrix_store(
            project_storage,
            pd.DataFrame.from_dict(data_dict),
            metadata,
        )
        results = uniform_distribution(
            db_engine,
            model_id=model.model_id,
            as_of_date=datetime.date(2016, 1, 1),
            test_matrix_store=test_store,
            n_ranks=5,
        )

        assert len(results) == 5  # 5 features x 1 entity for this as_of_date
        for result in results:
            assert "entity_id" in result
            assert "feature_name" in result
            assert "score" in result
            assert "feature_value" in result
            assert result["feature_value"] == 0.5
            assert result["score"] >= 0
            assert result["score"] <= 1
            assert isinstance(result["feature_name"], str)
            assert result["entity_id"] in [1, 2]
