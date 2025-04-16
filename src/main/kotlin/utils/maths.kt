// utils/maths.kt
package utils

import kotlin.math.*

// -------------------------------------------------------------------------------------
// interpolateData: Bilinear interpolation 함수
// -------------------------------------------------------------------------------------
/**
 * Bilinear interpolation을 수행하여 출력 이미지를 생성합니다.
 *
 * @param input 입력 이미지 데이터 (FloatArray) – 길이는 in_channels * in_height * in_width 이어야 함
 * @param inputDims Triple(in_channels, in_height, in_width)
 * @param outputDims Pair(out_height, out_width)
 * @param mode 현재 "bilinear"만 지원 (미사용)
 * @param alignCorners 미사용
 * @return FloatArray – 길이가 out_height * out_width * in_channels 인 배열
 */
fun interpolateData(
    input: FloatArray,
    inputDims: Triple<Int, Int, Int>,
    outputDims: Pair<Int, Int>,
    mode: String = "bilinear",
    alignCorners: Boolean = false
): FloatArray {
    val (inChannels, inHeight, inWidth) = inputDims
    val (outHeight, outWidth) = outputDims

    val xScale = outWidth.toFloat() / inWidth
    val yScale = outHeight.toFloat() / inHeight

    val outImg = FloatArray(outHeight * outWidth * inChannels)
    val inStride = inHeight * inWidth
    val outStride = outHeight * outWidth

    for (i in 0 until outHeight) {
        for (j in 0 until outWidth) {
            val outOffset = i * outWidth + j

            val x = (j + 0.5f) / xScale - 0.5f
            val y = (i + 0.5f) / yScale - 0.5f

            var x1 = floor(x).toInt()
            var y1 = floor(y).toInt()
            val x2 = min(x1 + 1, inWidth - 1)
            val y2 = min(y1 + 1, inHeight - 1)
            x1 = max(x1, 0)
            y1 = max(y1, 0)

            val s = x - x1
            val t = y - y1

            val w1 = (1 - s) * (1 - t)
            val w2 = s * (1 - t)
            val w3 = (1 - s) * t
            val w4 = s * t

            val yStride = y1 * inWidth
            val xStride = y2 * inWidth
            val idx1 = yStride + x1
            val idx2 = yStride + x2
            val idx3 = xStride + x1
            val idx4 = xStride + x2

            for (k in 0 until inChannels) {
                val cOffset = k * inStride
                outImg[k * outStride + outOffset] =
                    w1 * input[cOffset + idx1] +
                            w2 * input[cOffset + idx2] +
                            w3 * input[cOffset + idx3] +
                            w4 * input[cOffset + idx4]
            }
        }
    }
    return outImg
}

// -------------------------------------------------------------------------------------
// permuteData: 차원 순서 변경 함수
// -------------------------------------------------------------------------------------
/**
 * 입력 배열의 차원을 axes에 따라 순서를 변경합니다.
 *
 * @param array 입력 데이터 (FloatArray)
 * @param dims 각 차원의 크기를 담은 IntArray
 * @param axes 순서를 바꿀 축의 배열 (예: [2, 0, 1])
 * @return Pair(변환된 FloatArray, 새 shape를 담은 IntArray)
 */
fun permuteData(array: FloatArray, dims: IntArray, axes: IntArray): Pair<FloatArray, IntArray> {
    val nAxes = axes.size
    val shape = IntArray(nAxes)
    val stride = IntArray(nAxes)
    var s = 1
    for (i in nAxes - 1 downTo 0) {
        stride[i] = s
        shape[i] = dims[axes[i]]
        s *= shape[i]
    }
    // 각 원래 차원 i에 대해, axes에서의 위치에 해당하는 stride를 사용
    val invStride = IntArray(dims.size) { i ->
        val pos = axes.indexOf(i)
        if (pos >= 0) stride[pos] else 0
    }
    val permutedData = FloatArray(array.size)
    for (i in array.indices) {
        var newIndex = 0
        var k = i
        for (j in dims.size - 1 downTo 0) {
            newIndex += (k % dims[j]) * invStride[j]
            k /= dims[j]
        }
        permutedData[newIndex] = array[i]
    }
    return Pair(permutedData, shape)
}

// -------------------------------------------------------------------------------------
// 기타 수학 함수들
// -------------------------------------------------------------------------------------

/**
 * 주어진 배열(실수값 FloatArray)에서 최소값과 해당 인덱스를 반환합니다.
 * @param arr FloatArray
 * @return Pair(최소값, 최소값 인덱스)
 * @throws Exception 배열이 비어 있을 경우
 */
fun minValue(arr: FloatArray): Pair<Float, Int> {
    if (arr.isEmpty()) throw Exception("Array must not be empty")
    var minVal = arr[0]
    var indexOfMin = 0
    for (i in 1 until arr.size) {
        if (arr[i] < minVal) {
            minVal = arr[i]
            indexOfMin = i
        }
    }
    return Pair(minVal, indexOfMin)
}

/**
 * 주어진 배열(실수값 FloatArray)에서 최대값과 해당 인덱스를 반환합니다.
 * @param arr FloatArray
 * @return Pair(최대값, 최대값 인덱스)
 * @throws Exception 배열이 비어 있을 경우
 */
fun maxValue(arr: FloatArray): Pair<Float, Int> {
    if (arr.isEmpty()) throw Exception("Array must not be empty")
    var maxVal = arr[0]
    var indexOfMax = 0
    for (i in 1 until arr.size) {
        if (arr[i] > maxVal) {
            maxVal = arr[i]
            indexOfMax = i
        }
    }
    return Pair(maxVal, indexOfMax)
}

/**
 * 주어진 Float를 소수점 아래 decimals 자리까지 반올림합니다.
 *
 * @param num 반올림할 Float
 * @param decimals 반올림할 소수점 자릿수
 * @return 반올림된 Float
 */
fun round(num: Float, decimals: Int): Float {
    val pow = 10.0.pow(decimals.toDouble()).toFloat()
    return (kotlin.math.round(num * pow)) / pow
}

/**
 * Bankers' rounding (짝수 반올림): 소수점 이하 0.5인 경우 짝수로 반올림합니다.
 * 예: 1.5 → 2, 2.5 → 2
 *
 * @param x 반올림할 Float
 * @return Banker’s rounding 결과
 */
fun bankersRound(x: Float): Float {
    val r = round(x, 0)  // 소수점 없이 반올림한 값 (Float)
    return if (abs(x) % 1f == 0.5f) {
        if (r % 2f == 0f) r else r - 1f
    } else {
        r
    }
}

/**
 * 주어진 FloatArray에 softmax 함수를 적용합니다.
 *
 * @param arr FloatArray
 * @return softmax가 적용된 FloatArray
 */
fun softmax(arr: FloatArray): FloatArray {
    val (maxVal, _) = maxValue(arr)
    val exps = arr.map { exp((it - maxVal).toDouble()) }  // Double로 계산
    val sumExps = exps.sum()
    return exps.map { (it / sumExps).toFloat() }.toFloatArray()
}

/**
 * 주어진 FloatArray에 log_softmax 함수를 적용합니다.
 *
 * @param arr FloatArray
 * @return log_softmax가 적용된 FloatArray
 */
fun logSoftmax(arr: FloatArray): FloatArray {
    val (maxVal, _) = maxValue(arr)
    var sumExps = 0.0
    for (x in arr) {
        sumExps += exp((x - maxVal).toDouble())
    }
    val logSum = ln(sumExps)
    return arr.map { (it - maxVal - logSum).toFloat() }.toFloatArray()
}

/**
 * 두 FloatArray의 내적(dot product)을 계산합니다.
 *
 * @param arr1 첫 번째 FloatArray
 * @param arr2 두 번째 FloatArray
 * @return 내적 결과 (Double)
 */
fun dot(arr1: FloatArray, arr2: FloatArray): Double {
    var result = 0.0
    for (i in arr1.indices) {
        result += arr1[i] * arr2[i]
    }
    return result
}

/**
 * 두 FloatArray 사이의 코사인 유사도를 계산합니다.
 *
 * @param arr1 첫 번째 FloatArray
 * @param arr2 두 번째 FloatArray
 * @return 코사인 유사도 (Double)
 */
fun cosSim(arr1: FloatArray, arr2: FloatArray): Double {
    val dotProduct = dot(arr1, arr2)
    val magnitudeA = magnitude(arr1)
    val magnitudeB = magnitude(arr2)
    return dotProduct / (magnitudeA * magnitudeB)
}

/**
 * 주어진 FloatArray의 Euclidean norm(크기)를 계산합니다.
 *
 * @param arr FloatArray
 * @return 배열의 크기 (Double)
 */
fun magnitude(arr: FloatArray): Double {
    return sqrt(arr.fold(0.0) { acc, v -> acc + (v.toDouble() * v.toDouble()) })
}

/**
 * Dynamic Time Warping 알고리즘을 사용해 두 2차원 배열(행렬) 간 최적 정렬 경로(인덱스 시퀀스)를 구합니다.
 *
 * @param matrix 2차원 Double 배열
 * @return Pair(텍스트 인덱스 시퀀스, 시간 인덱스 시퀀스)
 */
fun dynamicTimeWarping(matrix: Array<DoubleArray>): Pair<List<Int>, List<Int>> {
    val outputLength = matrix.size
    val inputLength = matrix[0].size
    val outputRows = outputLength + 1
    val outputCols = inputLength + 1

    val cost = Array(outputRows) { DoubleArray(outputCols) { Double.POSITIVE_INFINITY } }
    cost[0][0] = 0.0

    val trace = Array(outputRows) { IntArray(outputCols) { -1 } }

    for (j in 1 until outputCols) {
        for (i in 1 until outputRows) {
            val c0 = cost[i - 1][j - 1]
            val c1 = cost[i - 1][j]
            val c2 = cost[i][j - 1]
            val (c, t) = when {
                c0 < c1 && c0 < c2 -> Pair(c0, 0)
                c1 < c0 && c1 < c2 -> Pair(c1, 1)
                else -> Pair(c2, 2)
            }
            cost[i][j] = matrix[i - 1][j - 1] + c
            trace[i][j] = t
        }
    }

    for (i in 0 until outputCols) {
        trace[0][i] = 2
    }
    for (i in 0 until outputRows) {
        trace[i][0] = 1
    }

    var i = outputLength
    var j = inputLength
    val textIndices = mutableListOf<Int>()
    val timeIndices = mutableListOf<Int>()
    while (i > 0 || j > 0) {
        textIndices.add(i - 1)
        timeIndices.add(j - 1)
        when (trace[i][j]) {
            0 -> { i--; j-- }
            1 -> i--
            2 -> j--
            else -> throw Exception("Internal error in dynamic time warping. Unexpected trace[$i, $j]. Please file a bug report.")
        }
    }
    textIndices.reverse()
    timeIndices.reverse()
    return Pair(textIndices, timeIndices)
}
