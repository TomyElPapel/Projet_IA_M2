

namespace AIPack.Package.Attributes;

[AttributeUsage(AttributeTargets.Field)]
public class Input : Attribute
{
    public string Name { get; }

    public Input(string name = "")
    {
        Name = name;
    }
}
