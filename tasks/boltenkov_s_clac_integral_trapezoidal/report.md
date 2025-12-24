# Отчёт по задаче: Максимальное значение элементов матрицы

**Национальный исследовательский Нижегородский государственный университет им. Н.И. Лобачевского**  
Институт информационных технологий, математики и механики

| | |
|------|------|
| **Направление подготовки** | Прикладная математика и информатика |
| **Вариант задания** | №8 |
| **Студент** | Болтенков С.С. (группа 3823Б1ПМоп3) |
| **Преподаватель** | доцент Сысоев А.В. |
| **Год выполнения** | 2025 |

## Введение


## Постановка задачи


## Описание алгоритма


## Описание схемы параллельного алгоритма


## Описание MPI-версии


## Результаты экспериментов и подтверждение корректности


## Выводы из результатов


## Заключение



## Список литературы
1. Документация в формате веб-сайта по реализации **MPICH** стандарта **MPI**: [https://www.mpich.org](https://www.mpich.org)


## Параллельная реализация
```cpp
  double BoltenkovSCalcIntegralkMPI::CalcCoef(const std::vector<double> &args,
                                            const std::vector<std::pair<double, double>> &limits) {
  double coef = 1.0;
  for (std::size_t i = 0; i < args.size(); i++) {
    if (std::abs(args[i] - limits[i].first) > 1e-14 && std::abs(args[i] - limits[i].second) > 1e-14) {
      coef *= 2;
    }
  }
  return coef;
}

void BoltenkovSCalcIntegralkMPI::CalcPoints(const int &n, const int &ind_cur_args, const std::vector<double> &h,
                                            std::unique_ptr<std::queue<std::vector<double>>> &args) {
  std::vector<double> cur_args;
  std::unique_ptr<std::queue<std::vector<double>>> args_tmp = std::make_unique<std::queue<std::vector<double>>>();
  while (!args->empty()) {
    cur_args = args->front();
    args->pop();
    args_tmp->push(cur_args);
    for (int step = 0; step < n; step++) {
      cur_args[ind_cur_args] += h[ind_cur_args];
      args_tmp->push(cur_args);
    }
    cur_args[ind_cur_args] += h[ind_cur_args];
  }
  std::swap(args, args_tmp);
}

double BoltenkovSCalcIntegralkMPI::calcIntegral(const int &n, const int &cnt_limits,
                                                const std::vector<std::pair<double, double>> &limits,
                                                double (*func)(std::vector<double>)) {
  double per = 1.0;
  std::vector<double> h(cnt_limits);
  for (std::size_t i = 0; i < h.size(); i++) {
    h[i] = (limits[i].second - limits[i].first) / static_cast<double>(n);
    per *= h[i];
  }

  std::vector<double> args_init(cnt_limits);
  for (std::size_t i = 0; i < args_init.size(); i++) {
    args_init[i] = limits[i].first;
  }

  std::unique_ptr<std::queue<std::vector<double>>> args = std::make_unique<std::queue<std::vector<double>>>();
  args->push(args_init);

  double res = 0.0;
  int ind_cur_args = 0;

  while (ind_cur_args != cnt_limits) {
    CalcPoints(n, ind_cur_args, h, args);
    ind_cur_args++;
  }

  while (!args->empty()) {
    args_init = args->front();
    args->pop();
    double coef = CalcCoef(args_init, limits);
    res += func(args_init) * coef * per / std::pow(2., static_cast<double>(cnt_limits));
  }

  res *= sign_integral_;
  return res;
}

bool BoltenkovSCalcIntegralkMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_Bcast(&sign_integral_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int n, cnt_limits;
  std::vector<std::pair<double, double>> limits;

  if (rank == 0) {
    n = std::get<0>(GetInput());
    cnt_limits = std::get<1>(GetInput());
    limits = std::get<2>(GetInput());
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cnt_limits, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    limits.resize(cnt_limits);
  }

  for (int i = 0; i < cnt_limits; i++) {
    double first, second;
    if (rank == 0) {
      first = limits[i].first;
      second = limits[i].second;
    }
    MPI_Bcast(&first, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&second, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank != 0) {
      limits[i] = {first, second};
    }
  }

  double local_a, local_b;
  double global_a = limits[0].first;
  double global_b = limits[0].second;
  double interval_length = (global_b - global_a) / size;

  local_a = global_a + rank * interval_length;
  local_b = global_a + (rank + 1) * interval_length;

  std::vector<std::pair<double, double>> local_limits = limits;
  local_limits[0] = {local_a, local_b};

  double local_integral = calcIntegral(n, cnt_limits, local_limits, std::get<3>(GetInput()));

  double &global_integral = GetOutput();
  MPI_Reduce(&local_integral, &global_integral, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  return true;
}
```