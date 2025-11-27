
using Microsoft.ML.Data;

namespace  IAPack.Package.MlDotnet;

public class DynamicRow
{
    [VectorType(4)]
    public float[] Features { get; set; } = [];
    public float Output { get; set; }
}