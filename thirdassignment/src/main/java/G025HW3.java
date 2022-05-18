import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class G025HW3 {
    public static void main(String[] args) {

        if (args.length != 4) {
            throw new IllegalArgumentException("USAGE: filepath k z L");
        }

        // ----- Initialize variables
        String filename = args[0];
        int k = Integer.parseInt(args[1]);
        int z = Integer.parseInt(args[2]);
        int L = Integer.parseInt(args[3]);

        // ----- Set Spark Configuration
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);
        SparkConf conf = new SparkConf(true).setAppName("MR k-center with outliers");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // ----- Read points from file
        long start = System.currentTimeMillis();
        JavaRDD<Vector> inputPoints = sc.textFile(filename, L)
                .map(G025HW3::strToVector)
                .repartition(L)
                .cache();
        long N = inputPoints.count();
        long end = System.currentTimeMillis();

        // ----- Print input parameters
        System.out.println("File : " + filename);
        System.out.println("Number of points N = " + N);
        System.out.println("Number of centers k = " + k);
        System.out.println("Number of outliers z = " + z);
        System.out.println("Number of partitions L = " + L);
        System.out.println("Time to read from file: " + (end - start) + " ms");

        // ---- Solve the problem
        ArrayList<Vector> solution = MR_kCenterOutliers(inputPoints, k, z, L);

        // ---- Compute the value of the objective function
        start = System.currentTimeMillis();
        double objective = computeObjective(inputPoints, solution, z);
        end = System.currentTimeMillis();
        System.out.println("Objective function = " + objective);
        System.out.println("Time to compute objective function: " + (end - start) + " ms");
    }

    private static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    private static double euclidean(Vector a, Vector b) {
        return Math.sqrt(Vectors.sqdist(a, b));
    }

    // Method MR_kCenterOutliers: MR algorithm for k-center with outliers
    private static ArrayList<Vector> MR_kCenterOutliers(JavaRDD<Vector> points, int k, int z, int L) {

        //------------- ROUND 1 ---------------------------
        long start = System.currentTimeMillis();
        JavaRDD<Tuple2<Vector, Long>> coreset = points.mapPartitions(x ->
        {
            ArrayList<Vector> partition = new ArrayList<>();
            while (x.hasNext())
                partition.add(x.next());
            ArrayList<Vector> centers = kCenterFFT(partition, k + z + 1);
            ArrayList<Long> weights = computeWeights(partition, centers);
            ArrayList<Tuple2<Vector, Long>> c_w = new ArrayList<>();
            for (int i = 0; i < centers.size(); i++) {
                Tuple2<Vector, Long> entry = new Tuple2<>(centers.get(i), weights.get(i));
                c_w.add(i, entry);
            }
            return c_w.iterator();
        }); // END OF ROUND 1
        long end = System.currentTimeMillis();
        System.out.println("Time to compute round 1: " + (end - start) + " ms");

        //------------- ROUND 2 ---------------------------
        // In Round 2, it collects the weighted coreset into a local data structure and runs method SeqWeightedOutliers,
        // "recycled" from Homework 2, to extract and return the final set of centers (you must fill in this latter part).
        ArrayList<Tuple2<Vector, Long>> elems = new ArrayList<>((k + z) * L);
        start = System.currentTimeMillis();
        elems.addAll(coreset.collect());
        end = System.currentTimeMillis();
        System.out.println("Time to compute round 2: " + (end - start) + " ms");
        //
        // ****** ADD YOUR CODE
        // ****** Compute the final solution (run SeqWeightedOutliers with alpha=2)
        // ****** Measure and print times taken by Round 1 and Round 2, separately
        // ****** Return the final solution
        List<Vector> vectors = elems.stream().map(x -> x._1).collect(Collectors.toList());
        List<Long> W = elems.stream().map(x -> x._2).collect(Collectors.toList());

        return SeqWeightedOutliers(vectors, W, k, z, 2);
    }

    private static ArrayList<Vector> kCenterFFT(ArrayList<Vector> points, int k) {

        final int n = points.size();
        double[] minDistances = new double[n];
        Arrays.fill(minDistances, Double.POSITIVE_INFINITY);

        ArrayList<Vector> centers = new ArrayList<>(k);

        Vector lastCenter = points.get(0);
        centers.add(lastCenter);

        for (int iter = 1; iter < k; iter++) {
            int maxIdx = 0;
            double maxDist = 0;

            for (int i = 0; i < n; i++) {
                double d = euclidean(points.get(i), lastCenter);
                if (d < minDistances[i]) {
                    minDistances[i] = d;
                }

                if (minDistances[i] > maxDist) {
                    maxDist = minDistances[i];
                    maxIdx = i;
                }
            }

            lastCenter = points.get(maxIdx);
            centers.add(lastCenter);
        }
        return centers;
    }

    private static ArrayList<Long> computeWeights(ArrayList<Vector> points, ArrayList<Vector> centers) {
        Long[] weights = new Long[centers.size()];
        Arrays.fill(weights, 0L);
        for (Vector point : points) {
            double tmp = euclidean(point, centers.get(0));
            int mycenter = 0;
            for (int j = 1; j < centers.size(); ++j) {
                if (euclidean(point, centers.get(j)) < tmp) {
                    mycenter = j;
                    tmp = euclidean(point, centers.get(j));
                }
            }
            // System.out.println("Point = " + points.get(i) + " Center = " + centers.get(mycenter));
            weights[mycenter] += 1L;
        }
        return new ArrayList<>(Arrays.asList(weights));
    }

    private static ArrayList<Vector> SeqWeightedOutliers(List<Vector> P, List<Long> W, int k, int z, int alpha) {
        final int PSIZE = P.size();
        double[][] distances = new double[PSIZE][PSIZE];

        final int LIMIT = k + z + 1;
        double r = Double.MAX_VALUE;
        for (int i = 0; i < PSIZE; i++) {
            distances[i][i] = 0; //Always 0 as distance
            for (int j = i + 1; j < PSIZE; j++) {
                double distance = euclidean(P.get(i), P.get(j));
                distances[j][i] = distance;
                distances[i][j] = distance;
                if (i < LIMIT && j < LIMIT)
                    r = Math.min(r, distance);
            }
        }
        r = r / 2;
        System.out.println("Initial guess = " + r);

        int GUESSES_ATTEMPT = 1;
        final long WOriginal = W.stream().reduce(0L, Long::sum);
        final List<Integer> ZOriginal = IntStream.range(0, PSIZE).boxed().collect(Collectors.toList());
        while (true) {
            ArrayList<Vector> S = new ArrayList<>();
            List<Integer> Z = new ArrayList<>(ZOriginal);
            long Wz = WOriginal;

            while (S.size() < k && Wz > 0) {
                long max = -1;
                int new_center_index = 0;

                for (int pointIndex = 0; pointIndex < PSIZE; pointIndex++) {
                    long ball_weight = FilterByRadius(Z, (1 + 2 * alpha) * r, distances[pointIndex])
                            .mapToLong(W::get)
                            .sum();

                    if (ball_weight > max) {
                        max = ball_weight;
                        new_center_index = pointIndex;
                    }
                }
                S.add(P.get(new_center_index));
                List<Integer> new_center_point = FilterByRadius(Z, (3 + 4 * alpha) * r, distances[new_center_index])
                        .collect(Collectors.toList());

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
                GUESSES_ATTEMPT++;
            }
        }
    }

    private static Stream<Integer> FilterByRadius(List<Integer> Z, double radius, double[] distances) {
        return Z.stream()
                .filter(x -> distances[x] < radius);
    }

    private static double computeObjective(JavaRDD<Vector> points, ArrayList<Vector> centers, int z) {

        JavaPairRDD<Double, Vector> result = points
                .flatMapToPair(p -> {
                    List<Tuple2<Vector, Vector>> pairs = new ArrayList<>();
                    for (Vector v : centers) {
                        pairs.add(new Tuple2<>(p, v));
                    }
                    return pairs.iterator();
                })
                .groupByKey()
                .mapToPair(x -> {
                    double min = Double.MAX_VALUE;
                    for (Vector v : x._2) {
                        min = Math.min(min, Math.sqrt(Vectors.sqdist(x._1, v)));
                    }
                    return new Tuple2<>(min, x._1);
                });

        List<Tuple2<Double, Vector>> last = result.sortByKey().take((int) result.count() - z);
        return last.get(last.size() - 1)._1;


        /*List<Vector> inputPoints = points.collect();
        List<Double> results = new ArrayList<>();
        for (Vector x : inputPoints) {
            double min = Double.MAX_VALUE;
            for (Vector v : centers) {
                min = Math.min(min, Math.sqrt(Vectors.sqdist(x, v)));
            }
            results.add(min);
        }
        Collections.sort(results);
        return Collections.max(results.subList(0, results.size() - z));*/
    }
}