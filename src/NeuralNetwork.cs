

namespace IAPack.Package;


public class NeuralNetwork
{
    private readonly INeuralNetworkBackend _backend;
    private readonly Dictionary<string, ParameterType> _parameters;
    private readonly List<Action> _actions;


    private static float ConvertToFloat(ParameterType type, object value)
    {
        return type switch
        {
            ParameterType.Float => Convert.ToSingle(value),
            ParameterType.Double => (float)Convert.ToDouble(value),
            ParameterType.Integer => Convert.ToSingle(value),
            ParameterType.Boolean => ((bool)value) ? 1f : 0f,
            _ => 0f
        };
    }
    
    private Dictionary<string, float> ConvertToFloatParameters(Dictionary<string, object> input)
    {
        Dictionary<string, float> output = [];
        foreach (var (name, value) in input)
        {
            output[name] = ConvertToFloat(_parameters[name], value);
        }
        return output;
    }


    public NeuralNetwork(INeuralNetworkBackend backend, NeuralNetworkConfig config)
    {
        _backend = backend;
        _parameters = config.Params;
        _actions = config.Actions;

        DataSetUtils.ExtractFromJsonl(config.DataPath, "Output", out var inputs, out var outputs);

        _backend.Train(inputs, outputs);
    }

    public void SetAction(int index, Action newAction)
    {
        _actions[index] = newAction;
    }

    public void Action(Dictionary<string, object> parameters)
    {
        var prediction = _backend.Predict(ConvertToFloatParameters(parameters));

        int actionIndex = -1;
        float actionPredi = float.MinValue;
        for (int i = 0; i < prediction.Length; i++)
        {
            if (prediction[i] > actionPredi)
            {
                actionPredi = prediction[i];
                actionIndex = i;
            }
        }
        
        if (_actions.Count >= actionIndex)
        {
            return;
        }

        _actions[actionIndex]?.Invoke();
    }

    public float[] Predict(Dictionary<string, object> parameters)
    {
        return _backend.Predict(ConvertToFloatParameters(parameters));
    }
}