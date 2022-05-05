import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.LongStream;

import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.collection.immutable.Stream;


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

        long start = System.currentTimeMillis();
        Object[] results = SeqWeightedOutliers(inputPoints, weights, k, z, 0);
        long finish = System.currentTimeMillis();
        List<Vector> solution = (List<Vector>) results[3];
        double res = ComputeObjective(inputPoints, solution, z);
        /*
        Input size n = 15
        Number of centers k = 3
        Number of outliers z = 1
        Initial guess = 0.04999999999999999
        Final guess = 0.7999999999999998
        Number of guesses = 5
        Objective function = 1.562049935181331
        Time of SeqWeightedOutliers = 422
        */

        System.out.println("Input size n = " + inputPoints.size());
        System.out.println("Number of centers k = " + k);
        System.out.println("Number of outliers z = " + z);
        System.out.println("Initial guess = " + results[0]);
        System.out.println("Final guess = " + results[1]);
        System.out.println("Number of guesses = " + results[2]);
        System.out.println("Objective function = " + res);
        System.out.println("Time of SeqWeightedOutliers = " + (finish - start));
    }

    private static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i=0; i<tokens.length; i++) {
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


    private static Object[] SeqWeightedOutliers(ArrayList<Vector> P, List<Long> W, int k, int z, int alpha) {
        List<Vector> points = P.subList(0, k + z + 1);
        double r = Double.MAX_VALUE;
        double initial_guesses, final_guesses;
        int guesses_attempt = 1;


        for(int i = 0; i < points.size(); i++) {
            for(int j = i + 1; j < points.size(); j++) {
                double distance = Math.sqrt(Vectors.sqdist(points.get(i), points.get(j)));
                r = Math.min(r, distance);
            }
        }
        r = r/2;
        initial_guesses = r;

        while (true) {
            List<Vector> S = new ArrayList<>();
            List<Vector> Z = new ArrayList<>(P);
            long Wz = W.stream().mapToLong(Long::longValue).sum();

            while (S.size() < k && Wz > 0) {
                long max = -1;
                Vector new_center = Vectors.dense(0,0);
                for(Vector point : P) {
                    List<Vector> res = Bz(point, (1 + 2*alpha) * r, Z);
                    long ball_weight = res.stream().mapToLong(y -> W.get(P.indexOf(y))).sum();
                    if(ball_weight > max) {
                        max = ball_weight;
                        new_center = point;
                    }
                }
                S.add(new_center);

                for(Vector y : Bz(new_center, (3 + 4*alpha) * r, Z)) {
                    Z.remove(y);
                    Wz = Wz - W.get(P.indexOf(y));
                }
            }

            if(Wz <= z) {
                final_guesses = r;
                return new Object[] { initial_guesses, final_guesses, guesses_attempt, S };
            } else {
                r = 2 * r;
                guesses_attempt++;
            }
        }
    }

    private static List<Vector> Bz(Vector x, double radius, List<Vector>Z) {
        // BZ (x,r) = {y ∈ Z : d(x, y) ≤ r}.
        List<Vector> res = new ArrayList<>();
        for (Vector point : Z) {
            if(Math.sqrt(Vectors.sqdist(x, point)) < radius) {
                res.add(point);
            }
        }
        return res;
    }

    private static double ComputeObjective(ArrayList<Vector> inputPoints, List<Vector> solution, int z) {
        List<Double> results = new ArrayList<>();
        for(Vector x : inputPoints) {
            double max = Double.MAX_VALUE;
            for(Vector s : solution) {
                max = Math.min(max, Math.sqrt(Vectors.sqdist(x, s)));
            }
            results.add(max);
        }
        Collections.sort(results);
        return Collections.max(results.subList(0, results.size() - z));
    }
}