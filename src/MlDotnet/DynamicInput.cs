
using Microsoft.ML.Data;

namespace  IAPack.Package.MlDotnet;

public class DynamicInput
{
    [VectorType] 
    public List<float> Values { get; set; } = [];
    public float Label { get; set; }
}