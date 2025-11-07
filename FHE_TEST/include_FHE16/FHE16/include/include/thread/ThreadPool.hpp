#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

typedef void(*TaskFunction)(void*);

class ThreadPool {
public:
    explicit ThreadPool(size_t threads);
    ~ThreadPool();

    void enqueue(TaskFunction f, void* arg);
    void wait_until_empty();

private:
    std::vector<std::thread> workers;
    std::queue<std::pair<TaskFunction, void*>> tasks;

    std::mutex queue_mutex;
    std::condition_variable condition;
    std::condition_variable wait_condition;
    bool stop;
};

#endif



