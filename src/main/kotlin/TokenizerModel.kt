import tokenizer.Unigram
import utils.Callable

// 설정 객체의 타입 (필요에 따라 별도 JSON 파서나 data class로 대체 가능)
typealias Config = Map<String, Any?>

/**
 * TokenizerModel은 토크나이저 모델의 기본 기능을 제공하는 추상 클래스입니다.
 * Callable을 상속받아 객체를 함수처럼 호출할 수 있도록 합니다.
 *
 * 이 구현은 Unigram 모델만 사용함에 맞게 단순화되었습니다.
 */
abstract class TokenizerModel(val config: Config) : Callable() {

    // vocab: 단어 목록 (인덱스 순서대로 저장)
    var vocab: MutableList<String> = mutableListOf()

    // 토큰과 id 간의 매핑
    var tokensToIds: MutableMap<String, Int> = mutableMapOf()

    // unknown token 관련 값
    var unkTokenId: Int? = null
    var unkToken: String? = null

    // (옵션) 단어 끝에 붙는 접미사
    var endOfWordSuffix: String? = null

    // unknown 토큰들을 인코딩할 때 융합할지 여부 (기본 false)
    var fuseUnk: Boolean = (config["fuse_unk"] as? Boolean) ?: false

    companion object {
        /**
         * config 객체를 기반으로 Unigram 모델의 TokenizerModel을 생성합니다.
         *
         * Unigram 모델만 사용할 경우이므로, config.type이 "Unigram"이거나 vocab 필드가 존재하면
         * Unigram 클래스로 생성합니다.
         *
         * @param config 설정 객체 (예: { "type": "Unigram", "vocab": [...] , ... })
         * @param args 추가 인자; 첫 번째 인자는 반드시 Config (moreConfig)여야 합니다.
         * @return Unigram 모델 기반의 TokenizerModel 객체
         * @throws Exception 알 수 없는 타입일 경우 예외 발생
         */
        fun fromConfig(config: Config, vararg args: Any): TokenizerModel {
            if (config["type"] == "Unigram" || config.containsKey("vocab")) {
                if (args.isNotEmpty() && args[0] is Map<*, *>) {
                    @Suppress("UNCHECKED_CAST")
                    val moreConfig = args[0] as Config
                    return Unigram(config, moreConfig)
                } else {
                    throw Exception("moreConfig must be provided as the first extra argument for Unigram model")
                }
            }
            throw Exception("Unsupported TokenizerModel type for Unigram-only mode: ${config["type"]}")
        }
    }

    /**
     * Callable의 _call 메서드 구현.
     * 첫 번째 인자로 전달된 토큰 리스트를 인코딩한 후,
     * fuseUnk 옵션이 활성화되어 있다면 unknown 토큰 융합 로직을 적용합니다.
     *
     * @param args 첫 번째 인자는 반드시 List<String> (토큰 리스트)여야 합니다.
     * @return 인코딩(및 후처리)된 토큰 리스트
     * @throws Exception 인자 타입이 올바르지 않은 경우
     */
    override fun _call(vararg args: Any?): Any? {
        val tokens = args.getOrNull(0) as? List<String>
            ?: throw Exception("첫 번째 인자는 List<String> 타입이어야 합니다.")
        var encoded = encode(tokens)
        if (fuseUnk) {
            encoded = fuseUnk(encoded, tokensToIds, unkTokenId)
        }
        return encoded
    }

    /**
     * 주어진 토큰 리스트를 인코딩하여 토큰(또는 서브토큰) 리스트를 반환합니다.
     * 서브 클래스(Unigram)에서 반드시 구현해야 합니다.
     */
    abstract fun encode(tokens: List<String>): List<String>

    /**
     * 토큰 리스트를 받아 각각의 토큰을 ID로 변환합니다.
     *
     * @param tokens 변환할 토큰 리스트
     * @return 토큰 ID 리스트 (해당 토큰이 매핑에 없으면 unkTokenId 사용)
     */
    fun convertTokensToIds(tokens: List<String>): List<Int?> =
        tokens.map { token -> tokensToIds[token] ?: unkTokenId }

    /**
     * 토큰 ID 리스트를 받아 각각의 ID에 해당하는 토큰을 반환합니다.
     *
     * @param ids 토큰 ID 리스트
     * @return ID에 해당하는 토큰 리스트 (유효하지 않은 ID면 unkToken 사용)
     */
    fun convertIdsToTokens(ids: List<Int>): List<String?> =
        ids.map { id -> if (id in vocab.indices) vocab[id] else unkToken }
}

/**
 * fuseUnk 함수
 *
 * 주어진 토큰 배열에서 unknown 토큰(unkTokenId와 동일한)을 연속하여 붙여 하나의 토큰으로 융합합니다.
 * JavaScript의 로직과 동일하게 구현하였습니다.
 *
 * @param tokens 인코딩된 토큰 리스트
 * @param tokensToIds 토큰 → ID 매핑
 * @param unkTokenId unknown token ID
 * @return unknown 토큰이 융합된 새로운 토큰 리스트
 */
fun fuseUnk(tokens: List<String>, tokensToIds: Map<String, Int>, unkTokenId: Int?): List<String> {
    val fused = mutableListOf<String>()
    var i = 0
    while (i < tokens.size) {
        fused.add(tokens[i])
        if ((tokensToIds[tokens[i]] ?: unkTokenId) != unkTokenId) {
            i++
            continue
        }
        while (++i < tokens.size && (tokensToIds[tokens[i]] ?: unkTokenId) == unkTokenId) {
            val lastIndex = fused.lastIndex
            if (tokensToIds[fused[lastIndex]] != unkTokenId) {
                fused[lastIndex] += tokens[i]
            }
        }
    }
    return fused
}
