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
class Ptr {
    List<Float> x;
    int total;

    public Ptr(List<Float> x, int total) {
        this.x = x;
        this.total = total;
    }


    public List<Float> align(int size) {
        this.total += size;
        List<Float> ret = this.x.subList(0, size);
        this.x = this.x.subList(size, this.x.size());
        return ret;
    }
}
