
using Microsoft.ML;
using Tensorflow;

namespace IAPack.Package.MlDotnet;


public class MlDotnetBackend : INeuralNetworkBackend
{
    private readonly MLContext _ml = new(seed: 0);
    private ITransformer? _model;
    private int _featureCount = -1;

    public void Train(Dictionary<string, List<float>> data, List<float> prediction)
    {
        if (data.Count == 0 || prediction.Count == 0)
            throw new ArgumentException("Data or prediction is empty");

        int rowCount = prediction.Count;

        var dynamicData = new List<Dictionary<string, float>>();
        for (int i = 0; i < rowCount; i++)
        {
            var row = new Dictionary<string, float>();
            foreach (var kv in data)
            {
                row[kv.Key] = kv.Value[i];
            }
            row["Output"] = prediction[i];
            dynamicData.Add(row);
        }

        var dataView = _ml.Data.LoadFromEnumerable(dynamicData);

        var featureColumns = new List<string>();
        foreach (var col in data.Keys)
            featureColumns.Add(col);

        var pipeline = _ml.Transforms.Concatenate("Features", featureColumns.ToArray())
            .Append(_ml.Regression.Trainers.Sdca(labelColumnName: "Output", featureColumnName: "Features"));

        _model = pipeline.Fit(dataView);
    }

    public float[] Predict(Dictionary<string, float> parameters)
    {
        if (_model == null)
            throw new InvalidOperationException("Model not trained. Call Train(...) first.");

        if (parameters.Count != _featureCount)
            throw new ArgumentException($"Feature vector must be length {_featureCount}");

        var engine = _ml.Model.CreatePredictionEngine<DynamicInput, DynamicPrediction>(_model);
        var input = new DynamicInput { Values = [.. parameters.Values] };
        var pred = engine.Predict(input);
        return pred.Prediction;
    }
}