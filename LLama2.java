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
class LLama2 {

    public static int sample(List<Float> probabilities, int n) {
        // Sample index from probabilities, they must sum to 1
        Random random = new Random();
        float r = random.nextFloat();
        float cdf = 0.0f;
        for (int i = 0; i < n; i++) {
            cdf += probabilities.get(i);
            if (r < cdf) {
                return i;
            }
        }
        return n - 1; // In case of rounding errors
    }

    public  static int readInt(RandomAccessFile f) {
        try {
            byte[] bytes = new byte[4];
            f.read(bytes, 0, 4);
            return ((bytes[0] & 0xFF) |
                    ((bytes[1] & 0xFF) << 8) |
                    ((bytes[2] & 0xFF) << 16) |
                    ((bytes[3] & 0xFF) << 24));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static float readFloat(RandomAccessFile f) {
        byte[] buffer = new byte[4];
        try {
            f.read(buffer, 0, 4);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        int intBits = ((buffer[3] & 0xFF) << 24) | ((buffer[2] & 0xFF) << 16) | ((buffer[1] & 0xFF) << 8) | (buffer[0] & 0xFF);
        return Float.intBitsToFloat(intBits);

    }

    public static String readString(RandomAccessFile f, int len) {
        try {
            byte[] bytes = new byte[len];
            f.read(bytes, 0, len);
            String s =  new String(bytes, StandardCharsets.UTF_8);
            return s;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static void accum(List<Float> a, List<Float> b) {
        int length = Math.min(a.size(), b.size());
        for (int i = 0; i < length; i++) {
            a.set(i, a.get(i)+ b.get(i));
        }
    }

    public static void rmsnorm(List<Float> o, List<Float> xo, List<Float> weight, int size) {
        assert size == o.size();

        // Calculate sum of squares
        float ss = 0.0f;
        for (int i = 0; i < size; i++) {
            float x = xo != null ? xo.get(i) : o.get(i);
            ss += x * x;
        }


        ss /= o.size();
        ss += 1e-5f;
        ss = 1.0f / (float) Math.sqrt(ss);

        // Normalize and scale
        for (int j = 0; j < size; j++) {
            float x = xo != null ? xo.get(j) : o.get(j);
            o.set(j, weight.get(j) * ss * x);
        }
    }

    public static void softmax(List<Float> x) {
        // Find max value (for numerical stability)
        float maxVal = x.get(0);
        for (float x_i : x) {
            maxVal = Math.max(maxVal, x_i);
        }

        // Exp and sum
        float sum = 0.0f;
        for (int i = 0; i < x.size(); i++) {
            float v = (float) Math.exp(x.get(i) - maxVal);
            x.set(i, v);
            sum += v;
        }

        // Normalize
        for (int i = 0; i < x.size(); i++) {
            x.set(i,  x.get(i)/sum);
        }


    }

    public static int strLookup(String str, List<String> vocab) {
        return vocab.indexOf(str);
    }

    public static int argMax(List<Float> v) {
        int index = -1;
        float maxVal = Float.NEGATIVE_INFINITY;

        for (int i = 0; i < v.size(); i++) {
            if (v.get(i) > maxVal) {
                maxVal = v.get(i);
                index = i;
            }
        }

        return index;
    }

    public static void matmul(List<Float> xout, List<Float> x, List<Float> w, int n, int d) {
        // W (d,n) @ x (n,) -> xout (d,)
        // By far the most amount of time is spent inside this little function
        assert d == xout.size();
        assert n == x.size();

        for (int i = 0; i < d; i++) {
            float sum = 0.0f;
            for (int j = 0; j < n; j++) {
                sum += w.get(i * n + j) * x.get(j);
            }
            xout.set(i, sum);
        }
    }

    public static float dot(List<Float> q, List<Float> k) {
        assert q.size() == k.size();
        float result = 0.0f;
        for (int i = 0; i < q.size(); i++) {
            result += q.get(i) * k.get(i);
        }
        return result;
    }



    public static  List<Float> copyOfRange(List<Float> list, int from, int to) {
        if (from < 0 || from > list.size() || to < 0 || to > list.size() || from > to)
            throw new IllegalArgumentException("Illegal extraction bounds");
        return list.subList(from, to);
    }

    public static void transformer(int token, int pos, Config p, RunState s, TransformerWeights w) {
        // A few convenience variables
        List<Float> x = s.x;
        int dim = p.dim;
        int hiddenDim = p.hidden_dim;
        int headSize = dim / p.n_heads;

        // Copy the token embedding into x
        int tokenStart = token * dim;
        for (int i = 0; i < dim; i++) {
            x.set(i, w.token_embedding_table.get(tokenStart + i));
        }


        // Pluck out the "pos" row of freqCisReal and freqCisImag
        List<Float> freqCisRealRow = LLama2.copyOfRange(w.freq_cis_real, pos * headSize / 2, (pos + 1) * headSize / 2);
        List<Float>  freqCisImagRow = LLama2.copyOfRange(w.freq_cis_imag, pos * headSize / 2, (pos + 1) * headSize / 2);


        // Forward all the layers
        for (int l = 0; l < p.n_layers; l++) {
            // Attention RMSNorm

            rmsnorm(s.xb, x, LLama2.copyOfRange(w.rms_att_weight, l * dim, (l + 1) * dim), dim);




            // QKV matmuls for this position
            matmul(s.q, s.xb, LLama2.copyOfRange(w.wq, l * dim * dim, (l + 1) * dim * dim), dim, dim);
            matmul(s.k, s.xb, LLama2.copyOfRange(w.wk, l * dim * dim, (l + 1) * dim * dim), dim, dim);
            matmul(s.v, s.xb, LLama2.copyOfRange(w.wv, l * dim * dim, (l + 1) * dim * dim), dim, dim);

            // Apply RoPE rotation to the q and k vectors for each head
            for (int h = 0; h < p.n_heads; h++) {
                int qStart = h * headSize;
                int kStart = h * headSize;
                for (int i = 0; i < headSize; i += 2) {
                    float q0 = s.q.get(qStart + i);
                    float q1 = s.q.get(qStart + i + 1);
                    float k0 = s.k.get(kStart + i);
                    float k1 = s.k.get(kStart + i + 1);
                    float fcr = freqCisRealRow.get(i / 2);
                    float fci = freqCisImagRow.get(i / 2);
                    s.q.set(qStart + i, q0 * fcr - q1 * fci);
                    s.q.set(qStart + i + 1, q0 * fci + q1 * fcr);
                    s.k.set(kStart + i, k0 * fcr - k1 * fci);
                    s.k.set(kStart + i + 1, k0 * fci + k1 * fcr);
                }
            }

            // Save key, value at this time step (pos) to our kv cache
            List<Float> keyCacheRow = s.key_cache.get(l).get(pos);
            List<Float> valueCacheRow = s.value_cache.get(l).get(pos);

            for (int i=0; i< s.k.size();i++) {
                keyCacheRow.set(i, s.k.get(i));
            }
            for (int i=0; i< s.v.size();i++) {
                valueCacheRow.set(i, s.v.get(i));
            }

            // Multi-head attention. Iterate over all heads
            for (int h = 0; h < p.n_heads; h++) {
                // Get the query vector for this head
                int qStart = h * headSize;
                List<Float> q = s.q.subList(qStart, qStart + headSize);
                List<Float> xb = s.xb.subList(qStart, qStart + headSize);

                // Attention scores for this head
                List<Float> att = s.att.get(h);

                // Iterate over all timesteps, including the current one
                for (int t = 0; t <= pos; t++) {
                    // Get the key vector for this head and at this timestep
                    int kStart = h * headSize;
                    List<List<Float>> lthLayer = s.key_cache.get(l);
                    List<Float> thTimestep = lthLayer.get(t);
                    List<Float> k = thTimestep.subList(kStart, kStart+headSize);


                    // Calculate the attention score as the dot product of q and k
                    float score = dot(q, k) / (float) Math.sqrt(headSize);

                    // Save the score to the attention buffer
                    att.set(t, score);
                }

                // Softmax the scores to get attention weights, from 0..pos inclusively
                softmax(att.subList(0, pos+1));

                // Weighted sum of the values, store back into xb

                for(int i=0; i<headSize;i++) {
                    xb.set(i, (float) 0.0f);
                }



                for (int t = 0; t <= pos; t++) {
                    // Get the value vector for this head and at this timestep
                    int vStart = h * headSize;


                    List<List<Float>> lthLayer = s.value_cache.get(l);
                    List<Float> thTimestep = lthLayer.get(t);
                    List<Float> v = thTimestep.subList(vStart, vStart+headSize);





                    // Get the attention weight for this timestep

                    float a = att.get(t);


                    // Accumulate the weighted value into xb
                    for (int i = 0; i < headSize; i++) {
                        xb.set( i, xb.get(i) + a * v.get(i));
                    }
                }
            }



            // Final matmul to get the output of the attention
            matmul(s.xb2, s.xb, LLama2.copyOfRange(w.wo, l * dim * dim, (l + 1) * dim * dim), dim, dim);




            // Residual connection back into x
            accum(x, s.xb2);



            //System.out.println(x);

            // FFN RMSNorm
            rmsnorm(s.xb, x, LLama2.copyOfRange(w.rms_ffn_weight, l * dim, (l + 1) * dim), dim);

            // Now for FFN in PyTorch, we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // First calculate self.w1(x) and self.w3(x)
            matmul(s.hb, s.xb, LLama2.copyOfRange(w.w1, l * dim * hiddenDim, (l + 1) * dim * hiddenDim), dim, hiddenDim);
            matmul(s.hb2, s.xb, LLama2.copyOfRange(w.w3, l * dim * hiddenDim, (l + 1) * dim * hiddenDim), dim, hiddenDim);

            // F.silu; silu(x) = x * σ(x), where σ(x) is the logistic sigmoid
            for (int i = 0; i < hiddenDim; i++) {
                s.hb.set(i, s.hb.get(i) * 1.0f / (1.0f + (float) Math.exp(-s.hb.get(i))));
            }

            // Elementwise multiply with w3(x)
            for (int i = 0; i < hiddenDim; i++) {
                s.hb.set(i, s.hb.get(i) * s.hb2.get(i));
            }

            // Final matmul to get the output of the FFN
            matmul(s.xb, s.hb, LLama2.copyOfRange(w.w2, l * dim * hiddenDim, (l + 1) * dim * hiddenDim), hiddenDim, dim);

            // Residual connection
            accum(x, s.xb);
        }

        // Final RMSNorm
        rmsnorm(x, null, w.rms_final_weight, dim);

        // Classifier into logits

        matmul(s.logits, x, w.wcls, dim, p.vocab_size);

    }

    public static long timeInMs() {
        // Return time in milliseconds, for benchmarking the model speed
        return Instant.now().toEpochMilli();
    }


}
