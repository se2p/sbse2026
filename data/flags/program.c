#include <stdio.h>
#include <string.h>

#define N 8

int square(int x) { return x * x; }
int clamp(int x, int lo, int hi) { return x < lo ? lo : (x > hi ? hi : x); }
int max2(int a, int b) { return a > b ? a : b; }

void mat_mul(int a[N][N], int b[N][N], int out[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++)
                sum += a[i][k] * b[k][j];
            out[i][j] = clamp(sum, -1000, 1000);
        }
    }
}

void mat_transpose(int a[N][N], int out[N][N]) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            out[j][i] = a[i][j];
}

int mat_trace(int a[N][N]) {
    int total = 0;
    for (int i = 0; i < N; i++)
        total += square(a[i][i]);
    return total;
}

void bubble_sort(int *arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int tmp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = tmp;
            }
        }
    }
}

int array_sum(int *arr, int n) {
    int total = 0;
    for (int i = 0; i < n; i++) total += arr[i];
    return total;
}

int array_max(int *arr, int n) {
    int best = arr[0];
    for (int i = 1; i < n; i++) best = max2(best, arr[i]);
    return best;
}

void str_reverse(char *s) {
    int len = strlen(s);
    for (int i = 0; i < len / 2; i++) {
        char tmp = s[i];
        s[i] = s[len - 1 - i];
        s[len - 1 - i] = tmp;
    }
}

int str_count_char(const char *s, char c) {
    int count = 0;
    int len = strlen(s);
    for (int i = 0; i < len; i++)
        if (s[i] == c) count++;
    return count;
}

void str_to_upper(char *s) {
    int len = strlen(s);
    for (int i = 0; i < len; i++)
        if (s[i] >= 'a' && s[i] <= 'z')
            s[i] -= 32;
}

const char *describe_code(int code) {
    switch (code) {
        case 0: return "zero";
        case 1: return "one";
        case 2: return "two";
        case 3: return "three";
        case 4: return "four";
        case 5: return "five";
        case 6: return "six";
        case 7: return "seven";
        default: return "many";
    }
}

int fibonacci(int n) {
    int a = 0, b = 1;
    for (int i = 2; i <= n; i++) {
        int c = a + b;
        a = b;
        b = c;
    }
    return n < 2 ? n : b;
}

int is_prime(int n) {
    if (n < 2) return 0;
    for (int i = 2; i * i <= n; i++)
        if (n % i == 0) return 0;
    return 1;
}

int count_primes(int limit) {
    int count = 0;
    for (int i = 2; i < limit; i++)
        if (is_prime(i)) count++;
    return count;
}

int main(void) {
    int a[N][N], b[N][N], c[N][N];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            a[i][j] = (i + j) % 7;
            b[i][j] = (i * j) % 5;
        }

    mat_mul(a, b, c);
    mat_transpose(c, a);
    printf("trace=%d\n", mat_trace(a));

    int nums[10] = {5, 3, 8, 1, 9, 2, 7, 4, 6, 0};
    bubble_sort(nums, 10);
    printf("sum=%d max=%d\n", array_sum(nums, 10), array_max(nums, 10));

    char msg[] = "Hello, Search-Based Software Engineering!";
    str_to_upper(msg);
    str_reverse(msg);
    printf("%s count=%d\n", msg, str_count_char(msg, 'E'));

    for (int i = 0; i < 8; i++)
        printf("%s ", describe_code(i));
    printf("\n");

    printf("fib(20)=%d primes<100=%d\n", fibonacci(20), count_primes(100));

    return 0;
}
