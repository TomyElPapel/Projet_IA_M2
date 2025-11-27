
namespace IAPack.Package;


public class NeuralNetworkConfig
{
    public required Dictionary<string, ParameterType> Params { init; get; }
    public required List<Action> Actions { get; init; }
    public required string DataPath { get; init; }
}