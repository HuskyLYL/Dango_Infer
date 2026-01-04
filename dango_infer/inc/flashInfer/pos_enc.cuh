

namespace flashinfer 
{
    enum class PosEncodingMode 
    {
        // No rotary positional embeddings
        kNone = 0U,
        // Apply Llama-style rope.
        kRoPELlama = 1U,
        // Apply ALiBi bias
        kALiBi = 2U
    };
}