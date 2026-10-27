# Анализ существующих решений

Промышленные фильтры сочетают правила и обучение. SpamAssassin складывает баллы около 800 правил и байесовского теста, спамом считается письмо от 5 баллов (Cormack, 2008). Gmail в 2019 году добавил модели TensorFlow поверх правил и стал отсекать еще около 100 млн писем в день (Kumaran, 2019). На TREC Spam Track онлайн-логистическая регрессия и SVM на символьных 4-граммах обошли наивный Байес (Sculley, Wachman, 2007). В Spam-T5 на тех же четырех корпусах, что у нас, F1 у Flan-T5 0,974 против 0,956 у SVM, а заметно выигрывают трансформеры только на малой выборке (Labonne, Moran, 2023). Ложная блокировка дороже пропуска. При λ=999 наивный Байес на Ling-Spam ловит лишь 63% спама (Androutsopoulos et al., 2000). Поэтому мы взяли линейную модель на словах и символьных n-граммах с порогом под FPR не выше 0,5%, а трансформеры оставили для сравнения.

## Источники

- Androutsopoulos I. et al. An evaluation of Naive Bayesian anti-spam filtering. ECML 2000 workshop. [arXiv:cs/0006013](https://arxiv.org/abs/cs/0006013)
- Sculley D., Wachman G. Relaxed Online SVMs for Spam Filtering. SIGIR 2007. [DOI 10.1145/1277741.1277813](https://doi.org/10.1145/1277741.1277813)
- Cormack G. Email Spam Filtering: A Systematic Review. Foundations and Trends in IR, 2008. [DOI 10.1561/1500000006](https://doi.org/10.1561/1500000006)
- Labonne M., Moran S. Spam-T5: Benchmarking Large Language Models for Few-Shot Email Spam Detection, 2023. [arXiv:2304.01238](https://arxiv.org/abs/2304.01238)
- Kumaran N. Ridding Gmail of 100 million more spam messages with TensorFlow, 2019. [Google Workspace blog](https://workspace.google.com/blog/product-announcements/ridding-gmail-of-100-million-more-spam-messages-with-tensorflow)
