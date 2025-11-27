using System.Text.Json;

namespace IAPack.Package;


public class DataSetUtils
{
    public static void ExtractFromJsonl(string filePath, string outputColumn, out Dictionary<string, List<float>> inputs, out List<float> outputs)
    {
        inputs = new Dictionary<string, List<float>>();
        outputs = new List<float>();

        foreach (var line in File.ReadLines(filePath))
        {
            try
            {
                using var doc = JsonDocument.Parse(line);
                var root = doc.RootElement;

                foreach (var prop in root.EnumerateObject())
                {
                    string name = prop.Name;
                    var value = prop.Value;

                    if (name == outputColumn)
                    {
                        outputs.Add(value.ValueKind switch
                        {
                            JsonValueKind.Number => value.GetSingle(),
                            JsonValueKind.True => 1f,
                            JsonValueKind.False => 0f,
                            _ => throw new Exception($"Output type {value.ValueKind} not supported")
                        });
                    }
                    else
                    {
                        if (!inputs.ContainsKey(name))
                            inputs[name] = new();

                        inputs[name].Add(value.ValueKind switch
                        {
                            JsonValueKind.Number => value.GetSingle(),
                            JsonValueKind.True => 1f,
                            JsonValueKind.False => 0f,
                            _ => throw new Exception($"Input type {value.ValueKind} not supported")
                        });
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erreur à la lecture d'une ligne : {ex.Message}");
            }
        }

        Console.WriteLine($"Extraction terminée. {outputs.Count} lignes lues, {inputs.Count} colonnes détectées.");
    }
}