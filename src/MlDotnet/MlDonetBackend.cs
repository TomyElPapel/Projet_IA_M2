
using Microsoft.ML;
using Tensorflow;

namespace IAPack.Package.MlDotnet;


public class MlDotnetBackend : INeuralNetworkBackend
{
    private readonly MLContext _ml = new(seed: 0);
    private ITransformer? _model;
    private string[]? _featureNames;
    private int _featureCount = -1;

    public void Train(Dictionary<string, List<float>> data, List<float> prediction)
    {
        if (data == null) throw new ArgumentNullException(nameof(data));
        if (prediction == null) throw new ArgumentNullException(nameof(prediction));
        if (data.Count == 0) throw new ArgumentException("No feature columns provided.", nameof(data));

        _featureNames = [.. data.Keys.OrderBy(k => k)];
        _featureCount = _featureNames.Length;

        var rows = new List<DynamicRow>();
        for (int i = 0; i < prediction.Count; i++)
        {
            var features = new float[_featureCount];
            for (int f = 0; f < _featureCount; f++)
            {
                var colName = _featureNames[f];
                features[f] = data[colName][i];
            }

            rows.Add(new DynamicRow
            {
                Features = features,
                Output = prediction[i]
            });
        }

        var dataView = _ml.Data.LoadFromEnumerable(rows);

       var pipeline = _ml.Transforms.NormalizeMinMax("Features")
                .Append(_ml.Regression.Trainers.Sdca(labelColumnName: nameof(DynamicRow.Output), featureColumnName: "Features"));

        _model = pipeline.Fit(dataView);
    }

    public float[] Predict(Dictionary<string, float> parameters)
    {
        if (_model == null)
                throw new InvalidOperationException("Model not trained. Call Train(...) first.");

        if (parameters == null) throw new ArgumentNullException(nameof(parameters));
        if (_featureNames == null)
            throw new InvalidOperationException("Feature order unknown (internal error).");

        var features = new float[_featureCount];
        for (int f = 0; f < _featureCount; f++)
        {
            var name = _featureNames[f];
            if (!parameters.TryGetValue(name, out var val))
                throw new ArgumentException($"Missing feature '{name}' in parameters for prediction.");

            features[f] = val;
        }

        var engine = _ml.Model.CreatePredictionEngine<DynamicRow, DynamicPrediction>(_model);
        var input = new DynamicRow { Features = features };
        var pred = engine.Predict(input);
        return [pred.Predicted];
    }
}