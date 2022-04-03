import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Serializable;
import scala.Tuple2;

import java.util.*;

public class Main {
    public static void main(String[] args) {

        if(args.length != 4) {
            throw new IllegalArgumentException("USAGE: num_partitions file_path");
        }

        //Spark Setup
        SparkConf conf = new SparkConf(true).setAppName("Assignment 1");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");


        //Input reading
        int K = Integer.parseInt(args[0]);
        int H = Integer.parseInt(args[1]);
        String S = String.valueOf(args[2]);
        JavaRDD<String> rawData = sc.textFile(args[3]).repartition(K).cache();
        System.out.println("Number of rows = " + rawData.count());

        //      1             2           3          4           5           6          7          8
        //TransactionID,  ProductID, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
        JavaPairRDD<String, Integer> productCustomer = rawData
                .flatMapToPair(row -> {
                    String[] info = row.split(",");
                    ArrayList<Tuple2<Tuple2<String, Integer>, Integer>> pairs = new ArrayList<>();
                    if(Integer.parseInt(info[3]) > 0) {
                        if(S.equals("all") || S.equals(info[7])) {
                            pairs.add(new Tuple2<>(new Tuple2<>(info[1], Integer.parseInt(info[6])),0));
                        }
                    }
                    return pairs.iterator();
                })
                .groupByKey()
                .flatMapToPair((productCustomers) -> {
                    List<Tuple2<String, Integer>> pairs = new ArrayList<>();
                    /*Integer counter = 0;
                    for (Object i: productCustomers._2)
                        counter ++;*/
                    pairs.add(new Tuple2<>(productCustomers._1._1, ((Collection<?>) productCustomers._2).size())); //counter
                    return pairs.iterator();
                });
        System.out.println("Product-Customer Pairs = " + productCustomer.count());

        JavaPairRDD<String, Integer> productPopularity1 = productCustomer
                .mapPartitionsToPair(productCustomers -> {
                    HashMap<String, Integer> counts = new HashMap<>();
                    List<Tuple2<String, Integer>> pairs = new ArrayList<>();
                    while (productCustomers.hasNext()) {
                        Tuple2<String, Integer> customerID = productCustomers.next();
                        counts.put(customerID._1, 1+counts.getOrDefault(customerID._1, 0));
                    }

                    for(Map.Entry<String, Integer> e: counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }

                    return pairs.iterator();

                })
                .groupByKey()
                .mapValues(test -> {
                    int result = 0;
                    for (Integer integer : test) {
                        result += integer;
                    }
                    return result;
                });


        JavaPairRDD<String, Integer> productPopularity2 = productCustomer
                .mapToPair(productCustomers -> new Tuple2<>(productCustomers._1, 1))
                .reduceByKey(Integer::sum);


        if(H > 0) {
            System.out.println("Top " + H + " Products and their Popularities");
            for(Tuple2<String,Integer> el: productPopularity1.takeOrdered(H, new TupleComparator())) {
                System.out.print("Product " + el._1 + " Popularity: " + el._2 + "; ");
            }
            System.out.println();
        } else if(H == 0)
        {
            System.out.println("productPopularity1:");
            for(Tuple2<String,Integer> el: productPopularity1.sortByKey().collect()) {
                System.out.print("Product: " + el._1 + " Popularity: " + el._2 + "; ");
            }
            System.out.println();
            System.out.println("productPopularity2:");
            for(Tuple2<String,Integer> el: productPopularity2.sortByKey().collect()) {
                System.out.print("Product: " + el._1 + " Popularity: " + el._2 + "; ");
            }
            System.out.println();
        }
    }

    static class TupleComparator implements Comparator<Tuple2<String, Integer>>, Serializable
    {
        @Override
        public int compare(Tuple2<String, Integer> o1, Tuple2<String, Integer> o2) {
            return o2._2.compareTo(o1._2);
        }
    }
}
