import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;


public class G025HW2 {
    public static void main(String[] args) throws IOException {
        if (args.length != 3) {
            throw new IllegalArgumentException("USAGE: file_path k z");
        }
        //Input reading
        String path = String.valueOf(args[0]);
        int k = Integer.parseInt(args[1]);
        int z = Integer.parseInt(args[2]);

        ArrayList<Vector> inputPoints = readVectorsSeq(path);
        ArrayList<Long> weights = new ArrayList<>(Collections.nCopies(inputPoints.size(), 1L));

        System.out.println("Input size n = " + inputPoints.size());
        System.out.println("Number of centers k = " + k);
        System.out.println("Number of outliers z = " + z);

        long start = System.currentTimeMillis();
        List<Vector> solution = SeqWeightedOutliers(inputPoints, weights, k, z, 0);
        long finish = System.currentTimeMillis();
        double res = ComputeObjective(inputPoints, solution, z);

        System.out.println("Objective function = " + res);
        System.out.println("Time of SeqWeightedOutliers = " + (finish - start));
    }

    private static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    private static ArrayList<Vector> readVectorsSeq(String filename) throws IOException {
        Path path = Path.of(filename);
        if (Files.isDirectory(path)) {
            throw new IllegalArgumentException("readVectorsSeq is meant to read a single file.");
        }
        ArrayList<Vector> result = new ArrayList<>();
        Files.lines(path)
                .map(G025HW2::strToVector)
                .forEach(result::add);
        return result;
    }

    private static ArrayList<Vector> SeqWeightedOutliers(ArrayList<Vector> P, List<Long> W, int k, int z, int alpha) {
        double [][] weights = new double[P.size()][P.size()];

        for(int i = 0; i < P.size(); i++) {
            for(int j = 0; j < P.size(); j++) {
                weights[i][j] = Math.sqrt(Vectors.sqdist(P.get(i), P.get(j)));
            }
        }

        double r = Double.MAX_VALUE;
        for (int i = 0; i < k + z + 1; i++) {
            for (int j = i + 1; j < k + z + 1; j++) {
                r = Math.min(r, weights[i][j]);
            }
        }
        r = r / 2;
        System.out.println("Initial guess = " + r);

        int GUESSES_ATTEMPT = 1;
        final long WSUM = W.stream().reduce(0L, Long::sum);
        final List<Integer> ZOriginal = IntStream.range(0, P.size()).boxed().collect(Collectors.toList());
        while (true) {
            ArrayList<Vector> S = new ArrayList<>();
            List<Integer> Z = new ArrayList<>(ZOriginal);
            long Wz = WSUM;

            while (S.size() < k && Wz > 0) {
                long max = -1;
                int new_center_index = 0;

                for (int pointIndex = 0; pointIndex < P.size(); pointIndex++) {
                    long ball_weight = BZ(Z, (1 + 2 * alpha) * r, weights[pointIndex])
                            .mapToLong(W::get)
                            .sum();

                    if (ball_weight > max) { // Seleziono il punto che "copre" più punti
                        max = ball_weight;
                        new_center_index = pointIndex;
                    }
                }
                S.add(P.get(new_center_index));
                List<Integer> new_center_point = BZ(Z, (3 + 4 * alpha) * r, weights[new_center_index]).collect(Collectors.toList());

                for (Integer t : new_center_point) {
                    Z.remove(t);
                    Wz -= W.get(t);
                }
            }

            if (Wz <= z) {
                System.out.println("Final guess = " + r);
                System.out.println("Number of guesses = " + GUESSES_ATTEMPT);
                return S;
            } else {
                r = 2 * r;
                System.out.println("Updated guess = " + r);
                GUESSES_ATTEMPT++;
            }
        }
    }

    private static Stream<Integer> BZ(List<Integer> Z,
                                     double radius,
                                     double[] weights_selected_center) {
        return Z.parallelStream()
                .filter(x -> weights_selected_center[x] < radius);
    }

    private static double ComputeObjective(ArrayList<Vector> inputPoints, List<Vector> solution, int z) {
        List<Double> results = new ArrayList<>();
        for (Vector x : inputPoints) {
            double min = Double.MAX_VALUE;
            for (Vector s : solution) {
                min = Math.min(min, Math.sqrt(Vectors.sqdist(x, s)));
            }
            results.add(min);
        }
        Collections.sort(results);
        return Collections.max(results.subList(0, results.size() - z));
    }
}