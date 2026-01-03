# GAN Flowers: Генерация изображений цветов с помощью Generative Adversarial Networks

Проект по генерации реалистичных изображений цветов из датасета [Oxford 102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) с использованием модифицированной архитектуры GAN, вдохновленной StyleGAN и современными техниками нормализации и активации.

## 📋 Содержание

- [Описание проекта](#описание-проекта)
- [Архитектура моделей](#архитектура-моделей)
  - [Генератор](#генератор)
  - [Дискриминатор](#дискриминатор)
- [Технические детали](#технические-детали)
- [Структура проекта](#структура-проекта)
- [Использование](#использование)
- [Результаты](#результаты)

## Описание проекта

Проект реализует генеративную модель для создания изображений цветов размером 64×64 пикселей. Модель обучена на датасете Flower102, содержащем 102 различных класса цветов. Архитектура объединяет современные техники из различных исследований в области генеративных моделей для достижения стабильного обучения и высокого качества генерации.

## Архитектура моделей

### Генератор

Генератор построен на основе архитектуры, вдохновленной **StyleGAN** (Karras et al., 2019), с несколькими ключевыми модификациями:

#### Mapping Network
- Преобразует входной шум `z ∈ ℝ¹⁰⁰` в скрытое представление `w ∈ ℝ¹²⁸`
- Состоит из двух полносвязных слоев с LeakyReLU активацией
- Инициализирован с помощью Kaiming initialization для стабильности

#### Style Blocks
Основные строительные блоки генератора используют адаптивную нормализацию и современные техники активации:

1. **SwiGLU активация** (Shazeer, 2020) - замена стандартных активаций на SwiGLU (Swish-Gated Linear Unit):
   ```
   SwiGLU(x) = Swish(x₁) ⊙ x₂
   ```
   где `x` разделяется на две части по каналам, а `Swish(x) = x · sigmoid(x)`. Эта активация показала отличные результаты в больших языковых моделях и здесь адаптирована для генеративных задач.

2. **RMSNorm** (Root Mean Square Layer Normalization, Zhang & Sennrich, 2019) - альтернатива BatchNorm:
   ```
   RMSNorm(x) = γ · x / √(mean(x²) + ε)
   ```
   RMSNorm не зависит от статистики батча, что делает его более стабильным для генеративных моделей.

3. **AdaIN (Adaptive Instance Normalization)** (Huang & Belongie, 2017) - инъекция стиля:
   ```
   AdaIN(x, w) = σ(w) · (x - μ(x)) / σ(x) + μ(w)
   ```
   Позволяет контролировать стиль генерируемых изображений через параметры `w` из mapping network.

4. **Noise Injection** - добавление перцептуального шума для увеличения разнообразия генерации, как в StyleGAN.

5. **Skip Connections** - остаточные связи с обучаемым масштабирующим коэффициентом для стабильного градиентного потока.

#### Self-Attention механизм
Механизм самовнимания (Self-Attention) добавлен на разрешении 32×32, вдохновленный **SAGAN** (Zhang et al., 2019):
- Позволяет модели учитывать глобальные зависимости в изображении
- Использует learnable параметр `γ` для постепенного включения внимания в процессе обучения
- Реализует стандартную схему Query-Key-Value attention

#### PixelShuffle Upsampling
Используется **PixelShuffle** (Shi et al., 2016) для эффективного upsampling:
- ICNR (Initialized Convolution for Neural Representation) инициализация для предотвращения артефактов шахматной доски
- Learnable blur после upsampling для сглаживания артефактов

#### Выходной слой
Финальный слой преобразует признаки в RGB изображение с активацией Tanh для диапазона [-1, 1].

**Архитектурная схема генератора:**
```
z (100) → Mapping Network → w (128)
                              ↓
                    input_const (4×4×1024)
                              ↓
                    StyleBlock → 8×8×512
                              ↓
                    StyleBlock → 16×16×256
                              ↓
                    StyleBlock → 32×32×128
                              ↓
                    Self-Attention
                              ↓
                    PixelShuffle → 64×64×64
                              ↓
                    Conv → 64×64×3 (Tanh)
```

### Дискриминатор

Дискриминатор использует архитектуру с **Spectral Normalization** (Miyato et al., 2018) для стабилизации обучения:

#### Spectral Normalization
Все сверточные слои используют Spectral Normalization, которая ограничивает липшицеву константу сети:
```
W_SN = W / σ(W)
```
где `σ(W)` - максимальное сингулярное значение матрицы весов. Это техника из **SN-GAN** для предотвращения коллапса дискриминатора.

#### ResNet-like Skip Connections
Дискриминатор использует skip connections, аналогичные ResNet (He et al., 2016):
- Параллельные пути: основной путь (main) и skip connection
- Skip connections используют AvgPool2d для downsampling и 1×1 свертки для изменения числа каналов
- Это позволяет градиентам свободно течь через сеть

#### Прогрессивное downsampling
Архитектура постепенно уменьшает пространственное разрешение:
- 64×64 → 32×32 → 16×16 → 8×8 → 4×4 → 1×1
- Удваивает число каналов на каждом уровне: 3 → 64 → 128 → 256 → 512 → 1

**Архитектурная схема дискриминатора:**
```
Input (64×64×3)
    ↓
[Skip1: AvgPool + 1×1 Conv] + [Main1: Conv blocks] → 32×32×128
    ↓
[Skip2: AvgPool + 1×1 Conv] + [Main2: Conv blocks] → 16×16×512
    ↓
[Skip3: AvgPool + 1×1 Conv] + [Main3: Conv blocks] → 1×1×1
    ↓
Output (scalar)
```

## Технические детали

### Функция потерь
Используется **Wasserstein GAN (WGAN)** loss (Arjovsky et al., 2017):
- **Дискриминатор**: `L_D = -E[D(x_real)] + E[D(x_fake)]`
- **Генератор**: `L_G = -E[D(x_fake)]`

WGAN обеспечивает более стабильное обучение по сравнению с классическим GAN loss.

### Оптимизация
- **Оптимизатор**: Adam (Kingma & Ba, 2014)
- **Learning Rate**: 2×10⁻⁴
- **Beta1**: 0.0 (рекомендуется для WGAN)
- **Beta2**: 0.9
- **Learning Rate Schedule**: Линейное затухание с 250-й эпохи до 0

### Exponential Moving Average (EMA)
Для генератора используется EMA с коэффициентом затухания 0.995:
- Создает более стабильную версию генератора для инференса
- Улучшает качество генерируемых изображений
- Сохраняется отдельно как `gEMA_*`

### Инициализация весов
- Сверточные и линейные слои: нормальная инициализация с `μ=0, σ=0.02`
- Mapping Network: Kaiming initialization для LeakyReLU
- BatchNorm/InstanceNorm: нормальная инициализация с `μ=1, σ=0.02`

### Аугментация данных
В проекте присутствует файл `DiffAugment.py`, который представляет собой неудавшийся эксперимент с дифференцируемой аугментацией данных. В финальной версии обучения аугментация не используется (`policy=''`).

### Датасет
- **Oxford 102 Category Flower Dataset**
- Разрешение: 64×64 пикселей
- Нормализация: `mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]` (диапазон [-1, 1])
- Batch size: 128

## Структура проекта

```
GAN_flowers/
├── GAN_flowers.ipynb          # Основной ноутбук с кодом обучения
├── DiffAugment.py             # Неудавшийся эксперимент с аугментацией
├── cat_to_name.json           # Маппинг классов цветов
├── dataset/
│   ├── train/                 # Обучающая выборка (102 класса)
│   ├── valid/                 # Валидационная выборка
│   └── test/                  # Тестовая выборка
└── models/
    ├── generator/
    │   ├── 497                # Чекпоинт генератора (эпоха 497)
    │   └── gEMA_497           # EMA версия генератора
    └── discriminator/
        └── 497                # Чекпоинт дискриминатора (эпоха 497)
```

## Использование

### Загрузка предобученных моделей

```python
from GAN_flowers import Generator, Discriminator, load_models, EMA

# Инициализация моделей
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator(nz=100, ngf=64, nc=3).to(device)
netD = Discriminator(nc=3, ndf=64).to(device)
g_ema = EMA(netG, decay=0.995)

# Загрузка чекпоинтов
load_models(netG=netG, nameG=497, netD=netD, nameD=497)
load_models(netG=g_ema.get_model(), nameG='gEMA_497', netD=netD, nameD=497)
```

### Генерация изображений

```python
def generate_images(netG, n_images=4, device='cuda'):
    with torch.no_grad():
        noise = torch.randn(n_images, 100, device=device)
        images = netG(noise).detach().cpu()
    return images

# Генерация с использованием EMA модели (рекомендуется)
images = generate_images(g_ema.get_model(), n_images=16)
```

### Обучение с нуля

1. Убедитесь, что датасет находится в `dataset/train/`
2. Запустите все ячейки ноутбука `GAN_flowers.ipynb`
3. Модели будут сохраняться каждые 100 эпох в `models/`
4. Рекомендуется обучение на GPU для ускорения процесса

## Результаты

Модель обучена на 500 эпохах. Финальные чекпоинты сохранены на эпохе 497. Для генерации рекомендуется использовать EMA версию генератора (`gEMA_497`), которая обеспечивает более стабильное и качественное генерирование изображений.

## Ссылки на научные статьи

1. **StyleGAN**: Karras, T., Laine, S., & Aila, T. (2019). A Style-Based Generator Architecture for Generative Adversarial Networks. CVPR 2019.

2. **Spectral Normalization**: Miyato, T., et al. (2018). Spectral Normalization for Generative Adversarial Networks. ICLR 2018.

3. **WGAN**: Arjovsky, M., Chintala, S., & Bottou, L. (2017). Wasserstein Generative Adversarial Networks. ICML 2017.

4. **SAGAN**: Zhang, H., et al. (2019). Self-Attention Generative Adversarial Networks. ICML 2019.

5. **AdaIN**: Huang, X., & Belongie, S. (2017). Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization. ICCV 2017.

6. **SwiGLU**: Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv:2002.05202.

7. **RMSNorm**: Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization. NeurIPS 2019.

8. **PixelShuffle**: Shi, W., et al. (2016). Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network. CVPR 2016.

9. **ResNet**: He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR 2016.

10. **Adam**: Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. ICLR 2015.

---

**Примечание**: Проект создан в образовательных целях и демонстрирует применение современных техник в области генеративных моделей.

