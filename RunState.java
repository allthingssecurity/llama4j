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
class RunState {
    // current wave of activations
    public List<Float> x;        // activation at the current time stamp (dim,)
    public List<Float> xb;       // same, but inside a residual branch (dim,)
    public List<Float> xb2;      // an additional buffer just for convenience (dim,)
    public List<Float> hb;       // buffer for the hidden dimension in the ffn (hidden_dim,)
    public List<Float> hb2;      // buffer for the hidden dimension in the ffn (hidden_dim,)
    public List<Float> q;        // query (dim,)
    public List<Float> k;        // key (dim,)
    public List<Float> v;        // value (dim,)
    public List<List<Float>> att; // buffer for scores/attention values (n_heads, seq_len)
    public List<Float> logits;   // output logits

    // kv cache
    public List<List<List<Float>>> key_cache;   // (layer, seq_len, dim)
    public List<List<List<Float>>> value_cache; // (layer, seq_len, dim)

    // Constructor
    public RunState(Config p) {
        x = new ArrayList<>();
        xb = new ArrayList<>();
        xb2 = new ArrayList<>();
        hb = new ArrayList<>();
        hb2 = new ArrayList<>();
        q = new ArrayList<>();
        k = new ArrayList<>();
        v = new ArrayList<>();
        att = new ArrayList<>();
        logits = new ArrayList<>();
        key_cache = new ArrayList<>();
        value_cache = new ArrayList<>();

        for (int i = 0; i < p.dim; i++) {
            x.add(0.0f);
            xb.add(0.0f);
            xb2.add(0.0f);
            q.add(0.0f);
            k.add(0.0f);
            v.add(0.0f);
        }

        for (int i = 0; i < p.hidden_dim; i++) {
            hb.add(0.0f);
            hb2.add(0.0f);
        }

        for (int i = 0; i < p.n_heads; i++) {
            List<Float> attRow = new ArrayList<>();
            for (int j = 0; j < p.seq_len; j++) {
                attRow.add(0.0f);
            }
            att.add(attRow);
        }

        for (int i = 0; i < p.vocab_size; i++) {
            logits.add(0.0f);
        }

        for (int i = 0; i < p.n_layers; i++) {
            List<List<Float>> keyCacheLayer = new ArrayList<>();
            List<List<Float>> valueCacheLayer = new ArrayList<>();

            for (int j = 0; j < p.seq_len; j++) {
                List<Float> keyCacheRow = new ArrayList<>();
                List<Float> valueCacheRow = new ArrayList<>();
                for (int k = 0; k < p.dim; k++) {
                    keyCacheRow.add(0.0f);
                    valueCacheRow.add(0.0f);
                }
                keyCacheLayer.add(keyCacheRow);
                valueCacheLayer.add(valueCacheRow);
            }

            key_cache.add(keyCacheLayer);
            value_cache.add(valueCacheLayer);
        }
    }
}
