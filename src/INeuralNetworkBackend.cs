namespace IAPack.Package;

public interface INeuralNetworkBackend
{
    public void Train(Dictionary<string, List<float>> data, List<float> prediction);
    public float[] Predict(Dictionary<string, float> parameters);
}