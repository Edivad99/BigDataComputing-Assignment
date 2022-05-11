import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;


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
        HashMap<Vector, Integer> positions = new HashMap<>();
        for (int i = 0; i < P.size(); i++) {
            positions.put(P.get(i), i);
        }

        List<Vector> points = P.subList(0, k + z + 1);
        double r = Double.MAX_VALUE;
        int GUESSES_ATTEMPT = 1;

        for (int i = 0; i < points.size(); i++) {
            for (int j = i + 1; j < points.size(); j++) {
                double distance = getDistance(points.get(i), points.get(j));
                r = Math.min(r, distance);
            }
        }
        r = r / 2;
        System.out.println("Initial guess = " + r);

        final long WSUM = W.stream().mapToLong(Long::longValue).sum();
        while (true) {
            ArrayList<Vector> S = new ArrayList<>();
            Set<Vector> Z = new HashSet<>(P);
            long Wz = WSUM;

            while (S.size() < k && Wz > 0) {
                long max = -1;
                Vector new_center = Vectors.dense(0, 0);
                List<Tuple2<Vector, Long>> new_center_point = new ArrayList<>();

                for (Vector point : P) {
                    List<List<Tuple2<Vector, Long>>> res = Bz(point, (1 + 2 * alpha) * r, (3 + 4 * alpha) * r, Z, positions, W);

                    long ball_weight = res.get(0)
                            .stream()
                            .mapToLong(y -> y._2)
                            .sum();

                    if (ball_weight > max) { // Seleziono il punto che "copre" più punti
                        max = ball_weight;
                        new_center = point;
                        new_center_point = res.get(1);
                    }
                }
                S.add(new_center);

                for (Tuple2<Vector, Long> y: new_center_point) {
                    Z.remove(y._1);
                    Wz = Wz - y._2;
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

    private static List<List<Tuple2<Vector, Long>>> Bz(Vector x,
                                                          double radius,
                                                          double radius2,
                                                          Set<Vector> Z,
                                                          HashMap<Vector, Integer> positions,
                                                          List<Long> W) {
        // BZ (x,r) = {y ∈ Z : d(x, y) ≤ r}.
        List<Tuple2<Vector, Long>> list1 = new ArrayList<>();
        List<Tuple2<Vector, Long>> list2 = new ArrayList<>();

        Z.forEach(point -> {
            double distance = getDistance(x, point);
            if (distance < radius2) {
                long weight = W.get(positions.get(point));
                list2.add(new Tuple2<>(point, weight));

                if (distance < radius) {
                    list1.add(new Tuple2<>(point, weight));
                }
            }
        });
        List<List<Tuple2<Vector, Long>>> result = new ArrayList<>();
        result.add(list1);
        result.add(list2);
        return result;
    }

    private static double getDistance(Vector x, Vector y) {
        return Math.sqrt(Vectors.sqdist(x, y));
    }

    private static double ComputeObjective(ArrayList<Vector> inputPoints, List<Vector> solution, int z) {
        List<Double> results = new ArrayList<>();
        for (Vector x : inputPoints) {
            double min = Double.MAX_VALUE;
            for (Vector s : solution) {
                min = Math.min(min, getDistance(x, s));
            }
            results.add(min);
        }
        Collections.sort(results);
        return Collections.max(results.subList(0, results.size() - z));
    }
}