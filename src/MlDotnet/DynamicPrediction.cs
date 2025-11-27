using Microsoft.ML.Data;

namespace  IAPack.Package.MlDotnet;

public class DynamicPrediction
{
    [ColumnName("output")]
    public float[] Prediction {get; set;}
}