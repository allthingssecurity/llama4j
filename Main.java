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


// Press Shift twice to open the Search Everywhere dialog and type `show whitespaces`,
// then press Enter. You can now see whitespace characters in your code.
public class Main {

public static void main(String[] args) throws FileNotFoundException {
    List<String> arguments = new ArrayList<>();
    for (String arg : args) {
        arguments.add(arg);
    }

    if (arguments.size() < 1) {
        System.out.println("Usage: <checkpoint_file> [temperature] [steps] [prompt]");
        return;
    }

    String checkpoint = arguments.get(0);
    float temperature = arguments.size() >= 2 ? Float.parseFloat(arguments.get(1)) : 0.9f;
    int steps = arguments.size() >= 3 ? Integer.parseInt(arguments.get(2)) : 256;
    String prompt = arguments.size() >= 4 ? arguments.get(3) : "";



    RandomAccessFile rf = new RandomAccessFile(checkpoint, "r");

    try (FileChannel fileChannel = FileChannel.open(new File(checkpoint).toPath(), StandardOpenOption.READ)) {
        MappedByteBuffer mmap = fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, fileChannel.size());
        mmap.order(ByteOrder.LITTLE_ENDIAN);
        float[] data1 = new float[mmap.remaining() / 4];
        mmap.asFloatBuffer().get(data1);
        Config config = Config.load(rf);
        List<Float> data = new ArrayList<>();
        for(int i=0;i<data1.length;i++){
            if (i<7) {
                continue;
            }
            data.add(data1[i]);
        }


        System.out.println("Configuration: " + config);

        TransformerWeights weights = TransformerWeights.init(config, data);

        if (steps <= 0 || steps > config.seq_len) {
            steps = config.seq_len;
        }

        RandomAccessFile tokenFile = new RandomAccessFile("tokenizer.bin", "r");
        Tokenizer tokenizer = Tokenizer.load(tokenFile, config);
        RunState state = new RunState(config);

        List<Integer> promptTokens = !prompt.isEmpty() ? tokenizer.bpeEncode(prompt) : new ArrayList<>();


        long start = 0;
        int next;
        int token = 1;
        int pos = 0;
        System.out.println("<s>");
        while (pos < steps) {
            LLama2.transformer(token, pos, config, state, weights);
            if (pos < promptTokens.size()) {
                next = promptTokens.get(pos);
            } else {
                if (temperature == 0.0) {
                    next = LLama2.argMax(state.logits);
                } else {
                    for (int q = 0; q < config.vocab_size; q++) {
                        state.logits.set(q, state.logits.get(q)/temperature);
                    }
                    LLama2.softmax(state.logits.subList(0, config.vocab_size));
                    next = LLama2.sample(state.logits, config.vocab_size);
                }
            }

            String tokenStr = token == 1 && tokenizer.vocab.get(next).startsWith(" ")
                    ? tokenizer.vocab.get(next).substring(1)
                    : tokenizer.vocab.get(next);
            System.out.print(tokenStr);
            System.out.flush();

            token = next;
            pos++;
            if (start == 0) {
                start = System.currentTimeMillis();
            }
        }

        long end = System.currentTimeMillis();
        System.out.println("\nachieved tok/s: " + ((steps - 1) / (end - start)) * 1000.0);
    } catch (IOException e) {
        e.printStackTrace();
    }
}

}