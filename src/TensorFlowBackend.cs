using Tensorflow;
using static Tensorflow.Binding;

namespace IAPack.Package;

public class TensorFlowBackend : INeuralNetworkBackend
{

    private Graph graph;
    private Session sess;
    private Tensor xPlaceholder;
    private Tensor yPlaceholder;
    private Tensor yPred;
    private Operation trainOp;
    private int featureCount;

    private float[] featureMin;
    private float[] featureMax;

    public void Train(Dictionary<string, List<float>> data, List<float> prediction)
    {
        if (data == null || data.Count == 0)
                throw new ArgumentException("Data empty");
            if (prediction == null || prediction.Count == 0)
                throw new ArgumentException("Prediction empty");

            featureCount = data.Count;
            int rowCount = prediction.Count;

            // Construire le graph
            graph = tf.Graph().as_default();
            sess = tf.Session(graph);

            xPlaceholder = tf.placeholder(tf.float32, shape: new Shape(-1, featureCount), name: "x");
            yPlaceholder = tf.placeholder(tf.float32, shape: new Shape(-1, 1), name: "y");

            var W = tf.Variable(tf.random.normal(new int[] { featureCount, 1 }), name: "weights");
            var b = tf.Variable(tf.zeros(new int[] { 1 }), name: "bias");

            yPred = tf.add(tf.matmul(xPlaceholder, W), b);

            var loss = tf.reduce_mean(tf.square(yPred - yPlaceholder));
            var optimizer = tf.train.GradientDescentOptimizer(0.01f);
            trainOp = optimizer.minimize(loss);

            sess.run(tf.global_variables_initializer());

            // Préparer les données
            float[,] X = new float[rowCount, featureCount];
            float[,] Y = new float[rowCount, 1];

            featureMin = new float[featureCount];
            featureMax = new float[featureCount];

            var featureNames = new List<string>(data.Keys);

            for (int f = 0; f < featureCount; f++)
            {
                featureMin[f] = float.MaxValue;
                featureMax[f] = float.MinValue;
                foreach (var val in data[featureNames[f]])
                {
                    if (val < featureMin[f]) featureMin[f] = val;
                    if (val > featureMax[f]) featureMax[f] = val;
                }
            }


            for (int i = 0; i < rowCount; i++)
            {
                for (int f = 0; f < featureCount; f++)
                {
                    float val = data[featureNames[f]][i];
                    float min = featureMin[f];
                    float max = featureMax[f];
                    X[i, f] = (max - min) != 0 ? (val - min) / (max - min) : 0f;
                }
                Y[i, 0] = prediction[i];
            }

            // Entraîner
            for (int epoch = 0; epoch < 500; epoch++)
                sess.run(trainOp, new FeedItem(xPlaceholder, X), new FeedItem(yPlaceholder, Y));
    }

    public float[] Predict(Dictionary<string, float> parameters)
    {
        if (parameters.Count != featureCount)
                throw new ArgumentException($"Feature count mismatch, expected {featureCount}");

        float[,] X = new float[1, featureCount];
        int i = 0;
        foreach (var val in parameters.Values)
        {
            float min = featureMin[i];
            float max = featureMax[i];
            X[0, i] = (max - min) != 0 ? (val - min) / (max - min) : 0f;
            i++;
        }

        var result = sess.run(yPred, new FeedItem(xPlaceholder, X));
        return [result[0, 0]];
    }
}