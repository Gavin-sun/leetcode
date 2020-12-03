package Array;

/**
 * @author Gavin
 * @version 1.0
 * @date 2020 12 2020/12/3 17:13
 *
 *      题目:合并排序的数组 https://leetcode-cn.com/problems/sorted-merge-lcci
 *      给定两个排序后的数组 A 和 B，其中 A 的末端有足够的缓冲空间容纳 B。
 *      编写一个方法，将 B 合并入 A 并排序。
 *      初始化A 和 B 的元素数量分别为m 和 n。
 *      输入:
 *          A = [1,2,3,0,0,0], m = 3
 *          B = [2,5,6],       n = 3
 *
 *      输出: [1,2,2,3,5,6]
 *      提示: A.length == n + m
 *
 */
public class meet10_01 {

    public static void main(String[] args) {
        int A[]={1,2,3,0,0,0};
        int B[]={2,5,6};
        merge(A,3,B,3);
        for (int i : A) {
            System.out.print(i+" ");
        }
    }

    public static void merge(int[] A, int m, int[] B, int n){
        int id=m+n-1;
        int ai=m-1;
        int bi=n-1;
        while(bi>=0){
            if(ai>=0&&A[ai]>=B[bi]){
                A[id--]=A[ai--];
            }else{
                A[id--]=B[bi--];
            }
        }
    }
}
