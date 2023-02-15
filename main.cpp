#define _USE_MATH_DEFINES
#include <cmath>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <random>

struct intLine {
    long threadSum;
    int padding[14];
};

struct result {
    double time;
    double answer;
    double variation;

    result() {}

    result (double _time, double _answer, double _variation) {
        time = _time;
        answer = _answer;
        variation = _variation;
    }

    result (double _time, double r, int n, int count) {
       time = _time;
       answer = 4 * r * r * count / n;
       variation = answer / (M_PI * r * r);
    }
};

result parallel_solve(double r, long n, int threads_count, int chunk_size)
{
    double start = omp_get_wtime() * 1000; // Время начала выполнения тела метода

    // Константа, равная количеству различных чисел,
    // которые может выдать используемый генератор случайных чисел.
    unsigned long long const MAX = (1LL << 31) - 1;
    unsigned long long const MAXSQRT = MAX * MAX;

    // Оптимизация работы кэша
    struct intLine thread_sum[threads_count] __attribute__ ((aligned(64)));
    long count = 0;
    std::random_device rd;

    #pragma omp parallel default(none) num_threads(threads_count) shared(r, n, count, thread_sum, rd, MAX, MAXSQRT, chunk_size)
    {
        std::mt19937 gen(rd());

        int tid = omp_get_thread_num(); // Номер потока
        thread_sum[tid].threadSum = 0;

        #pragma omp for schedule(dynamic, chunk_size)
        for (long i = 0; i < n; ++i) {
            unsigned long long x = gen() >> 1;
            unsigned long long y = gen() >> 1;
            if (x * x + y * y <= MAXSQRT) {
                thread_sum[tid].threadSum += 1;
            }
        }
        #pragma omp atomic
        count += thread_sum[tid].threadSum;
    }
    double deltaTime = omp_get_wtime() * 1000 - start;
    return {deltaTime, r, n, count};
}


result usual_solve(double r, int n)
{
    double start = omp_get_wtime() * 1000;

    unsigned long long const MAX = (1LL << 31) - 1;
    unsigned long long const MAXSQRT = MAX * MAX;

    double const RS = r * r;
    int count = 0;

    std::random_device rd;
    std::mt19937 gen(rd());

    for (int i = 0; i < n; ++i) {
        unsigned long long x = gen() >> 1;
        unsigned long long y = gen() >> 1;
        if (x * x + y * y <= MAXSQRT) {
            ++count;
        }
    }
    double deltaTime = omp_get_wtime() * 1000 - start;
    return {deltaTime, r, n, count};
}


void test_solve(double r, int n, int threads_count, int test_count, int chunk_size, result* testes, bool isShow = false)
{
    double sumTime = 0;
    double sumAnswer = 0;
    double sumVariation = 0;

    if (threads_count == 0) {
        threads_count = omp_get_max_threads();
    }
    for (int i = 0; i < test_count; ++i) {
        if (threads_count == -1) {
            testes[i] = usual_solve(r, n);
        } else {
            testes[i] = parallel_solve(r, n, threads_count, chunk_size);
        }
        sumTime += testes[i].time;
        sumAnswer += testes[i].answer;
        sumVariation += testes[i].variation;
    }
    testes[test_count] = result(sumTime / test_count, sumAnswer / test_count, sumVariation / test_count);
    if (threads_count == -1) {
        threads_count = 0;
    }
    if (isShow) {
        printf("Time (%i thread(s)): %g ms\n", threads_count, sumTime / test_count);
    }
}

void write_results(result* testes, int test_count, double r, int n, std::string output, std::string debug) {

    std::ofstream deb(debug);
    if (deb.is_open()) {
        double maxTime = 0;
        double maxVariation = 0;
        double minVariation = 2;
        for (int i = 0; i < test_count; ++i) {
            maxTime = std::max(maxTime, testes[i].time);
            maxVariation = std::max(maxVariation, testes[i].variation);
            minVariation = std::min(minVariation, testes[i].variation);
            deb << i + 1 << ".\t" << testes[i].time << "ms\t" << testes[i].answer << "\t" << testes[i].variation << "\n";
        }

        deb << "\nResult:\n";
        deb << "\tAvgTime:\t\t" << testes[test_count].time << "ms\n";
        deb << "\tMaxTime:\t\t" << maxTime << "ms\n";
        deb << "\tAvgAnswer:\t\t" << testes[test_count].answer << "\n";
        deb << "\tAvgVariation:\t" << testes[test_count].variation << "\n";
        deb << "\tMaxVariation:\t" << maxVariation;
        deb << "\tMinVariation:\t" << minVariation;
    }

    std::ofstream out(output);
    if (out.is_open()) {
        out << testes[0].answer;
    } else {
        std::cout << "Output file can't be opened.";
    }
    out.close();
    deb.close();
}

int main(int argc, char** argv)
{
    try {
        double r;
        long n;
        int threads_count = std::stoi(argv[1]);
        std::string input = argv[2];
        std::string output = argv[3];
        std::string debug = "debug.txt";
        int test_count = 1;
        int chunk_size = 64;


        if (false) {
            std::cin >> debug;
        }

        std::ifstream in(input);
        if (in.is_open()) {
            in >> r >> n;
            result *testes = new result[test_count + 1];
            test_solve(r, n, threads_count, test_count, chunk_size, testes, true);
            write_results(testes, test_count, r, n, output, debug);
        } else {
            std::cout << "Input file can't be opened.";
        }
        in.close();
    } catch(std::invalid_argument const& ex) {
        std::cout << "Incorrect number of threades: " << ex.what() << '\n';
    } catch(const std::ifstream::failure& e) {
        std::cout << "Problem with opening input file: " << e.what() << '\n';
    } catch(const std::ofstream::failure& e) {
        std::cout << "Problem with opening outnput file: " << e.what() << '\n';
    }
}
