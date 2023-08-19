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
class Tokenizer {
    public List<String> vocab;
    public List<Float> vocab_scores;
    public int max_token_length;

    public Tokenizer() {
        this.vocab = new ArrayList<>();
        this.vocab_scores = new ArrayList<>();
        this.max_token_length =0;
    }

    public static Tokenizer load(RandomAccessFile file, Config config) throws IOException {
        Tokenizer tokenizer = new Tokenizer();
        try  {
            tokenizer.max_token_length = LLama2.readInt(file);
            for (int i = 0; i < config.vocab_size; i++) {
                tokenizer.vocab_scores.add(LLama2.readFloat(file));
                int len =  LLama2.readInt(file);
                tokenizer.vocab.add(LLama2.readString(file, len));
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return tokenizer;
    }





    public List<Integer> bpeEncode(String text) {
        List<Integer> tokens = new ArrayList<>();
        StringBuilder strBuffer = new StringBuilder();

        // First encode every individual byte in the input string
        for (int i = 0; i < text.length(); i++) {
            strBuffer.setLength(0); // Clear the StringBuilder
            strBuffer.append(text.charAt(i));
            int id = LLama2.strLookup(strBuffer.toString(), vocab);
            if (id != -1) {
                tokens.add(id);
            } else {
                throw new IllegalArgumentException("Token not found in vocab.");
            }
        }

        // Merge the best consecutive pair each iteration, according to the scores in vocab_scores
        while (true) {
            double bestScore = -1e10;
            int bestId = 0;
            Integer bestIdx = null;

            for (int i = 0; i < (tokens.size() - 1); i++) {
                // Check if we can merge the pair (tokens[i], tokens[i+1])
                strBuffer.setLength(0); // Clear the StringBuilder
                strBuffer.append(vocab.get(tokens.get(i)));
                strBuffer.append(vocab.get(tokens.get(i + 1)));
                int id = LLama2.strLookup(strBuffer.toString(), vocab);
                if (id != -1) {
                    if (vocab_scores.get(id) > bestScore) {
                        // This merge pair exists in vocab! Record its score and position
                        bestScore = vocab_scores.get(id);
                        bestId = id;
                        bestIdx = i;
                    }
                }
            }

            if (bestIdx == null) {
                return tokens; // We couldn't find any more pairs to merge, so we're done
            } else {
                // Merge the consecutive pair (bestIdx, bestIdx+1) into new token bestId
                tokens.set(bestIdx, bestId);
                // Delete token at position bestIdx+1, shift the entire sequence back 1
                for (int i = (bestIdx + 1); i < (tokens.size() - 1); i++) {
                    tokens.set(i, tokens.get(i + 1));
                }
                tokens.remove(tokens.size() - 1); // Token length decreased
            }
        }
    }

}