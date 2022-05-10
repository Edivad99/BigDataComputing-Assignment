import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

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


    private static List<Vector> SeqWeightedOutliers(ArrayList<Vector> P, List<Long> W, int k, int z, int alpha) {
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
            List<Vector> S = new ArrayList<>();
            Set<Vector> Z = new HashSet<>(P);
            long Wz = WSUM;

            while (S.size() < k && Wz > 0) {
                long max = -1;
                Vector new_center = Vectors.dense(0, 0);
                for (Vector point : P) {
                    List<Vector> res = Bz(point, (1 + 2 * alpha) * r, Z);
                    long ball_weight = res.stream().mapToLong(y -> W.get(P.indexOf(y))).sum();
                    if (ball_weight > max) {
                        max = ball_weight;
                        new_center = point;
                    }
                }
                S.add(new_center);

                for (Vector y : Bz(new_center, (3 + 4 * alpha) * r, Z)) {
                    Z.remove(y);
                    Wz = Wz - W.get(P.indexOf(y));
                }
            }

            if (Wz <= z) {
                System.out.println("Final guess = " + r);
                System.out.println("Number of guesses = " + GUESSES_ATTEMPT);
                return S;
            } else {
                r = 2 * r;
                GUESSES_ATTEMPT++;
            }
        }
    }

    //static HashMap<Double, List<Vector>> hashMap = new HashMap<>();
    private static List<Vector> Bz(Vector x, double radius, Set<Vector> Z) {
        // BZ (x,r) = {y ∈ Z : d(x, y) ≤ r}.
        //if(hashMap.containsKey(radius))
        //    return hashMap.get(radius);

        List<Vector> res = new ArrayList<>();
        for (Vector point : Z) {
            if (getDistance(x, point) < radius) {
                res.add(point);
            }
        }
        //hashMap.put(radius, res);
        return res;
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

    static HashMap<String, Double> map = new HashMap<>();

    private static double getDistance(Vector x, Vector y) {
        //String pair1 = x.toString() + "," + y.toString();
        //String pair2 = y.toString() + "," + x.toString();

        /*if (map.containsKey(pair1)) {
            return map.get(pair1);
        } else if (map.containsKey(pair2)){
            return map.get(pair2);
        }*/

        double distance = Math.sqrt(Vectors.sqdist(x, y));
        //map.put(pair1, distance);
        return distance;
    }
}