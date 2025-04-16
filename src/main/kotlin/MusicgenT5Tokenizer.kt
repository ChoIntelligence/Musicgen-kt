package org.example.musicgen

import com.fasterxml.jackson.module.kotlin.*
import java.io.File
import kotlin.math.min

/**
 * Jackson을 이용해 tokenizer.json, tokenizer_config.json을 로드하고,
 * JS 코드에서의 AutoTokenizer.from_pretrained(...)와 비슷한 과정을 수행하는 예시.
 *
 * usage 예시:
 *
 *   val tokenizer = AutoTokenizer.fromPretrained("path/to/dir")
 *   val ids = tokenizer.encode("Hello MusicGen!")
 *   println(ids)  // JS와 동일한 인코딩 결과가 나오는지 확인
 */
object AutoTokenizer {
    /**
     * fromPretrained: JS의 AutoTokenizer.from_pretrained와 비슷한 로직
     * @param modelPath  tokenizer.json, tokenizer_config.json이 존재하는 폴더 경로
     */
    fun fromPretrained(
        modelPath: String
    ): T5Tokenizer {
        // 1) tokenizer.json, tokenizer_config.json 파일 로드
        val tokenizerJsonFile = File(modelPath, "tokenizer.json")
        val tokenizerConfigFile = File(modelPath, "tokenizer_config.json")

        require(tokenizerJsonFile.exists()) { "Cannot find tokenizer.json at $modelPath" }
        require(tokenizerConfigFile.exists()) { "Cannot find tokenizer_config.json at $modelPath" }

        // 2) Jackson으로 파싱
        val mapper = jacksonObjectMapper()
        val tokenizerJson = mapper.readValue<TokenizerJson>(tokenizerJsonFile)
        val tokenizerConfig = mapper.readValue<TokenizerConfig>(tokenizerConfigFile)

        // 3) T5Tokenizer(혹은 필요한 PreTrainedTokenizer) 생성
        //    실제 JS쪽 'tokenizer_class'가 'T5Tokenizer'인 경우.
        //    => musicgen-small 에서는 T5Tokenizer 기반(실은 Unigram)으로 알려짐
        //    => 여기서는 T5Tokenizer를 반환
        return T5Tokenizer(tokenizerJson, tokenizerConfig)
    }
}

/**
 * tokenizer.json 에 대한 Jackson 매핑용 데이터 클래스 예시
 * (필요한 필드만 발췌: Unigram 토크나이저에 필요한 model / normalizer 등)
 */
data class TokenizerJson(
    val model: ModelConfig,
    val normalizer: NormalizerConfig? = null,
    val pre_tokenizer: PreTokenizerConfig? = null,
    val post_processor: PostProcessorConfig? = null,

    // 아래는 단순화된 예시
    val added_tokens: List<AddedTokenConfig>? = null,
    // ...
)

/**
 * tokenizer_config.json 에 대한 Jackson 매핑용 예시
 * (주요 속성만)
 *
 * T5Tokenizer를 예로 들면 보통:
 *   "tokenizer_class": "T5Tokenizer"
 *   "unk_token": "<unk>"
 *   "eos_token": "</s>"
 *   "pad_token": "<pad>"
 *   "model_max_length": 512
 * 등이 들어있음
 */
data class TokenizerConfig(
    val tokenizer_class: String? = null,
    val model_max_length: Int? = 512,
    val unk_token: String? = "<unk>",
    val eos_token: String? = "</s>",
    val pad_token: String? = "<pad>",
    val clean_up_tokenization_spaces: Boolean? = true,
    // 필요시 추가
)

/** tokenizer.json의 "model" 필드를 위한 Jackson용 데이터 클래스 */
data class ModelConfig(
    val type: String?,
    val unk_id: Int? = null,
    val vocab: List<List<Any>>? = null,     // Unigram인 경우 [[piece, score], [piece, score], ...]
    // ...
)

/** Normalizer/PreTokenizer/PostProcessor 등은 MusicGen용 T5라면 크게 안 쓸 수도 있음. 필요하면 구현. */
data class NormalizerConfig(val type: String?)
data class PreTokenizerConfig(val type: String?)
data class PostProcessorConfig(val type: String?)

/** added_tokens: special token 목록 */
data class AddedTokenConfig(
    val content: String,
    val id: Int,
    val special: Boolean? = false,
    // ...
)

/**
 * JS의 "T5Tokenizer"에 대응되는 클래스.
 * 내부적으로 Unigram 모델을 이용해 encode/convertTokens/convertIds 를 수행.
 */
class T5Tokenizer(
    private val tokenizerJson: TokenizerJson,
    private val tokenizerConfig: TokenizerConfig
) {
    private val model: UnigramModel
    private val unkToken: String
    private val eosToken: String
    private val padToken: String

    private val maxLength: Int
    private val cleanUpSpaces: Boolean

    init {
        // -- tokenizer_config.json에서 special token들 추출
        unkToken = tokenizerConfig.unk_token ?: "<unk>"
        eosToken = tokenizerConfig.eos_token ?: "</s>"
        padToken = tokenizerConfig.pad_token ?: "<pad>"

        maxLength = tokenizerConfig.model_max_length ?: 512
        cleanUpSpaces = tokenizerConfig.clean_up_tokenization_spaces ?: true

        // -- tokenizer.json > model 설정이 "Unigram"이라고 가정 (Musicgen)
        //    vocab: 이중 리스트 [[string, score], [string, score], ...]
        //    unk_id: ?
        //    ...
        require(tokenizerJson.model.type == "Unigram") {
            "현재 예시는 Unigram(T5)만 가정. type=${tokenizerJson.model.type}"
        }

        val vocabList = tokenizerJson.model.vocab
            ?: error("Unigram에 필요한 vocab 정보가 tokenizer.json에 없음")

        val unkId = tokenizerJson.model.unk_id
            ?: error("Unigram에 필요한 unk_id가 없음")

        model = UnigramModel(
            vocabList.mapIndexed { i, (pieceAny, scoreAny) ->
                val piece = pieceAny.toString()  // 보통 string
                val score = scoreAny.toString().toFloatOrNull() ?: 0f
                piece to score
            },
            unkId = unkId
        )
    }

    /**
     * 사용 예시:
     *   val tokens = tokenizer.tokenize("Hello MusicGen!")
     *   println(tokens)
     */
    fun tokenize(text: String): List<String> {
        // 1) (Optional) normalizer/ pre_tokenizer 처리
        //    musicgen tokenizer.json 상 normalizer가 간단하거나 null일 수도 있음
        val normalized = text // 여기서는 단순히 text 그대로. 필요시 구현

        // 2) unigram (sentencepiece 기반)으로 tokenize
        //    -> model.encode() 호출
        //    -> piece 단위로 반환
        return model.encode(normalized)
    }

    /**
     * encode: 토큰을 id로 변환
     */
    fun encode(text: String): List<Int> {
        val tokens = tokenize(text)
        return convertTokensToIds(tokens)
    }

    /**
     * 여러 문장 입력 시, (JS처럼) batch-encode 가능 예시
     */
    fun encodeBatch(texts: List<String>): List<List<Int>> {
        return texts.map { encode(it) }
    }

    /**
     * id -> token
     */
    fun convertIdsToTokens(ids: List<Int>): List<String> {
        return ids.map { id -> model.idToPiece(id) }
    }

    /**
     * token -> id
     */
    fun convertTokensToIds(tokens: List<String>): List<Int> {
        return tokens.map { token -> model.pieceToId(token) }
    }

    /**
     * 간단한 디코딩(문자열 복원). 실제 T5는 부분적으로 처리 로직이 복잡할 수 있으나,
     * 여기서는 아주 간단히 piece들을 이어붙이거나 필요시 postProcess.
     */
    fun decode(ids: List<Int>): String {
        // 1) id -> piece
        val pieces = convertIdsToTokens(ids)

        // 2) piece 들을 단순 join
        //    T5 default: 공백 없이 이어붙은 문자열을, sentencepiece decode 규칙으로 복원
        //    여기서는, 시범적으로 아래와 같은 단순 decode:
        val sb = StringBuilder()
        for (p in pieces) {
            // skip <unk> <pad> <s> </s> 등도 가능(원하면)
            if (p == unkToken || p == padToken || p == eosToken) {
                // skip special tokens
                continue
            }
            sb.append(p)
        }
        val raw = sb.toString()

        // 3) clean up (tokenization spaces 등)
        return if (cleanUpSpaces) {
            // 예: 공백 전후 trim 등...
            raw.replace("▁", " ").trim()
        } else {
            raw
        }
    }
}

/**
 * 실제 Unigram 모델 (SentencePiece style) 최소 구현 예
 *  - T5 등의 tokenizer.json 내부 'model': 'Unigram' + vocab
 *  - vocab: (piece, score) 리스트
 *  - unkId: 정해진 토큰 id
 *
 * 주의: 실제 sentencepiece의 세부 로직(특히 "lattice", "sample" 등)은 간단치 않지만,
 * 여기서는 "encode"를 단순히 (전체 문자열 -> piece들) 나누는 로직 정도로 예시화함.
 * JS쪽 tokenizer.js에서는 utils.TokenLattice 활용 + trie 기반 search 등을 구현하고 있으므로
 * **아래는 매우 간략화**된 버전입니다. (실제 동일 결과를 원하면 trie, lattice 로직 전부 구현해야 함)
 */
class UnigramModel(
    private val vocab: List<Pair<String, Float>>,
    private val unkId: Int
) {
    // piece->id, id->piece
    private val pieceToIdMap: Map<String, Int>
    private val idToPieceMap: Map<Int, String>

    init {
        pieceToIdMap = buildMap {
            vocab.forEachIndexed { i, (piece, _) ->
                put(piece, i)
            }
        }
        idToPieceMap = vocab.mapIndexed { i, (piece, _) -> i to piece }.toMap()
    }

    fun pieceToId(piece: String): Int {
        return pieceToIdMap[piece] ?: unkId
    }

    fun idToPiece(id: Int): String {
        return idToPieceMap[id] ?: idToPieceMap[unkId] ?: "<unk>"
    }

    /**
     * JS쪽 true Unigram tokenize를 완벽히 재현하려면:
     *  - utils.TokenLattice, Trie, Viterbi 등을 구현
     *  - "모든 가능한 sub-piece"를 시도해서 가장 점수 높은 경로를 찾는 식
     *
     * 여기서는 간단히 "▁"를 공백 대용으로 보고, 띄어쓰기 단위로 나눈 뒤,
     * vocab에 있으면 그 토큰을 그대로, 없으면 unk 등으로 처리하는 **단순 버전** 예시
     */
    fun encode(text: String): List<String> {
        // 아주 단순히 " " 기준 split -> 각각 "▁토큰" 형태로. (실제로는 수많은 잡다한 규칙이 필요)
        val chunks = text.split("\\s+".toRegex()).filter { it.isNotBlank() }
        val resultPieces = mutableListOf<String>()

        for (chunk in chunks) {
            // 실제 sentencepiece 에서는 chunk 내에서 더 작은 서브토큰 단위로 분리
            // 아래는 그냥 "▁" + chunk 로 vocab 탐색 시도하는 예시
            val spmPiece = if (!chunk.startsWith("▁")) "▁$chunk" else chunk

            if (pieceToIdMap.containsKey(spmPiece)) {
                resultPieces.add(spmPiece)
            } else {
                // 없는 경우, 그냥 unk 처리
                resultPieces.add(idToPieceMap[unkId] ?: "<unk>")
            }
        }

        return resultPieces
    }
}

