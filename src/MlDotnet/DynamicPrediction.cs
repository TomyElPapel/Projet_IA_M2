using Microsoft.ML.Data;

namespace  IAPack.Package.MlDotnet;

public class DynamicPrediction
{
    [ColumnName("Score")]
    public float Predicted { get; set; }
}