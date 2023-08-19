import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.lang.reflect.Field;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.*;
class TransformerWeights {
    // token embedding table
    public List<Float> token_embedding_table; // (vocab_size, dim)
    // weights for rmsnorms
    public List<Float> rms_att_weight; // (layer, dim) rmsnorm weights
    public List<Float> rms_ffn_weight; // (layer, dim)
    // weights for matmuls
    public List<Float> wq; // (layer, dim, dim)
    public List<Float> wk; // (layer, dim, dim)
    public List<Float> wv; // (layer, dim, dim)
    public List<Float> wo; // (layer, dim, dim)
    // weights for ffn
    public List<Float> w1; // (layer, hidden_dim, dim)
    public List<Float> w2; // (layer, dim, hidden_dim)
    public List<Float> w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    public List<Float> rms_final_weight; // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    public List<Float> freq_cis_real; // (seq_len, dim/2)
    public List<Float> freq_cis_imag; // (seq_len, dim/2)
    // (optional) classifier weights for the logits, on the last layer
    public List<Float> wcls;

    public TransformerWeights() {
        token_embedding_table = new ArrayList<>();
        rms_att_weight = new ArrayList<>();
        rms_ffn_weight = new ArrayList<>();
        wq = new ArrayList<>();
        wk = new ArrayList<>();
        wv = new ArrayList<>();
        wo = new ArrayList<>();
        w1 = new ArrayList<>();
        w2 = new ArrayList<>();
        w3 = new ArrayList<>();
        rms_final_weight = new ArrayList<>();
        freq_cis_real = new ArrayList<>();
        freq_cis_imag = new ArrayList<>();
        wcls = new ArrayList<>();
    }

    // Constructor
    public static TransformerWeights init(Config p, List<Float> f) {
        TransformerWeights ret = new TransformerWeights();
        int head_size = p.dim / p.n_heads;

        Ptr ptr = new Ptr(f, 0);
        ret.token_embedding_table = ptr.align(p.vocab_size * p.dim);
        ret.rms_att_weight = ptr.align(p.n_layers * p.dim);
        ret.wq = ptr.align(p.n_layers * p.dim * p.dim);
        ret.wk = ptr.align(p.n_layers * p.dim * p.dim);
        ret.wv = ptr.align(p.n_layers * p.dim * p.dim);
        ret.wo = ptr.align(p.n_layers * p.dim * p.dim);
        ret.rms_ffn_weight = ptr.align(p.n_layers * p.dim);
        ret.w1 = ptr.align(p.n_layers * p.hidden_dim * p.dim);
        ret.w2 = ptr.align(p.n_layers * p.dim * p.hidden_dim);
        ret.w3 = ptr.align(p.n_layers * p.hidden_dim * p.dim);
        ret.rms_final_weight = ptr.align(p.dim);
        ret.freq_cis_real = ptr.align(p.seq_len * head_size / 2);
        ret.freq_cis_imag = ptr.align(p.seq_len * head_size / 2);

        if (!p.shared_weight) {
            ret.wcls = ptr.align(p.dim * p.vocab_size);
        } else {
            ret.wcls = ret.token_embedding_table;
        }

        assert ptr.total == f.size();
        return ret;
    }
}
