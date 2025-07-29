enum class DType
{
    float16,
    float32,
    int8,
    int32,
    uint8
};

inline size_t DtypeToSize(DType dtype)
{
    switch (dtype)
    {
    case DType::float32:
        return 4;
    case DType::float16:
        return 2;
    case DType::int32:
        return 4;
    case DType::int8:
        return 1;
    case DType::uint8:
        return 1;
    }

    return 0;
}