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
class  Config {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
    boolean shared_weight;

    public static Config load(RandomAccessFile randomAccessFile) throws IOException {
        try  {
            Config conf = new Config();
            conf.dim = LLama2.readInt(randomAccessFile);
            conf.hidden_dim = LLama2.readInt(randomAccessFile);
            conf.n_layers = LLama2.readInt(randomAccessFile);
            conf.n_heads = LLama2.readInt(randomAccessFile);
            conf.n_kv_heads = LLama2.readInt(randomAccessFile);

            int vocab_size = LLama2.readInt(randomAccessFile);
            conf.vocab_size = Math.abs(vocab_size);
            conf.shared_weight = vocab_size > 0;
            conf.seq_len = LLama2.readInt(randomAccessFile);

            return conf;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
    @Override
    public String toString()
    {
        StringBuilder result = new StringBuilder();
        String newLine = System.getProperty("line.separator");

        result.append( this.getClass().getName() );
        result.append( " Object {" );
        result.append(newLine);

        //determine fields declared in this class only (no fields of superclass)
        Field[] fields = this.getClass().getDeclaredFields();

        //print field names paired with their values
        for ( Field field : fields  ) {
            result.append("  ");
            try {
                result.append( field.getName() );
                result.append(": ");
                //requires access to private field:
                result.append( field.get(this) );
            } catch ( IllegalAccessException ex ) {
                System.out.println(ex);
            }
            result.append(newLine);
        }
        result.append("}");

        return result.toString();
    }
}
