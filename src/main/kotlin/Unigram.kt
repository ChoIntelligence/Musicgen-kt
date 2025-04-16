package tokenizer

import utils.min
import utils.Callable
import TokenizerModel
import utils.CharTrie
import utils.TokenLattice

// 설정 객체의 타입 (필요하면 별도 data class로 정의할 수 있습니다)
typealias Config = Map<String, Any?>

/**
 * Unigram 클래스는 Unigram 모델을 이용하여 토큰화를 수행하는 토크나이저 모델입니다.
 * TokenizerModel (Callable 상속)을 상속받으며, JS의 로직과 동일하게 구현되어 있습니다.
 */
class Unigram(config: Config, moreConfig: Config) : TokenizerModel(config) {
    // Unigram 모델 전용 멤버 변수
    var scores: MutableList<Double> = mutableListOf()
    lateinit var trie: CharTrie
    var minScore: Double = 0.0
    var unkScore: Double = 0.0

    // bos, eos 관련 변수
    var bosToken: String? = null
    var bosTokenId: Int? = null
    var eosToken: String? = null
    var eosTokenId: Int? = null

    init {
        // config["vocab"]는 각 원소가 [token, score] 형태의 List여야 함
        val vocabList = config["vocab"] as? List<*>
            ?: throw Exception("vocab must be provided and be a List")
        val vocabSize = vocabList.size

        // vocab과 scores 초기화
        this.vocab = MutableList(vocabSize) { "" }
        this.scores = MutableList(vocabSize) { 0.0 }
        for (i in 0 until vocabSize) {
            // 각 element는 List<Any?> 형태, 첫번째 원소는 token(String), 두번째는 score(Number)
            val piece = vocabList[i] as? List<*>
                ?: throw Exception("Each vocab piece must be a List")
            val token = piece.getOrNull(0) as? String
                ?: throw Exception("Token must be a String")
            val score = (piece.getOrNull(1) as? Number)?.toDouble()
                ?: throw Exception("Score must be a Number")
            this.vocab[i] = token
            this.scores[i] = score
        }

        // unk_token_id: config에서 지정된 숫자 (Int)
        this.unkTokenId = (config["unk_id"] as? Number)?.toInt()
            ?: throw Exception("unk_id must be provided")
        this.unkToken = this.vocab[this.unkTokenId!!]

        // vocab 배열을 순회하며 tokensToIds 맵 구성 (TokenizerModel의 tokensToIds 필드 사용)
        this.tokensToIds = mutableMapOf()
        for ((i, token) in this.vocab.withIndex()) {
            this.tokensToIds[token] = i
        }

        // bos 토큰은 공백(" ")으로 고정
        this.bosToken = " "
        this.bosTokenId = this.tokensToIds[this.bosToken]
        // moreConfig에서 eos_token 설정 (예: "</s>")
        this.eosToken = moreConfig["eos_token"] as? String
            ?: throw Exception("eos_token must be provided in moreConfig")
        this.eosTokenId = this.tokensToIds[this.eosToken]

        // unk 토큰 재설정 (필요시)
        this.unkToken = this.vocab[this.unkTokenId!!]

        // scores의 최소값과 그 인덱스를 구함 (min 함수는 Pair(minValue, minIndex) 반환)
        val (minVal, _) = min(this.scores)
        this.minScore = minVal

        // unkScore는 minScore - 10.0
        this.unkScore = this.minScore - 10.0
        // unk 토큰에 해당하는 score를 unkScore로 업데이트
        if (this.unkTokenId != null && this.unkTokenId!! < this.scores.size) {
            this.scores[this.unkTokenId!!] = this.unkScore
        } else {
            throw Exception("Invalid unk_token_id")
        }

        // CharTrie 초기화 후, vocab 전체를 추가
        this.trie = CharTrie()
        this.trie.extend(this.vocab)

        // Unigram 모델은 fuseUnk를 강제 true
        this.fuseUnk = true
    }

    /**
     * TokenLattice에 trie 결과를 바탕으로 노드를 추가합니다.
     *
     * @param lattice 토큰 격자 (TokenLattice)
     */
    fun populateNodes(lattice: TokenLattice) {
        val chars = lattice.chars
        val mblen = 1
        var beginPos = 0
        while (beginPos < chars.size) {
            var hasSingleNode = false
            // beginPos부터 문자열 슬라이스 생성
            val sliced = chars.subList(beginPos, chars.size).joinToString("")
            // trie에서 해당 슬라이스에 대해 공통 접두사를 검색 (Sequence<String>로 반환; toList()로 변환)
            val prefixedTokens = trie.commonPrefixSearch(sliced).toList()
            for (token in prefixedTokens) {
                val tokenId = tokensToIds[token] ?: continue
                val tokenScore = scores[tokenId]
                val n = token.length
                lattice.insert(beginPos, n, tokenScore, tokenId)
                if (!hasSingleNode && n == mblen) {
                    hasSingleNode = true
                }
            }
            if (!hasSingleNode) {
                lattice.insert(beginPos, mblen, unkScore, unkTokenId!!)
            }
            beginPos += mblen
        }
    }

    /**
     * 주어진 정규화된 문자열을 Unigram 모델을 사용하여 토큰화합니다.
     *
     * @param normalized 정규화된 문자열
     * @return 토큰화된 서브토큰 리스트
     */
    fun tokenize(normalized: String): List<String> {
        // TokenLattice 생성 시 bosTokenId와 eosTokenId를 사용
        val lattice = TokenLattice(normalized, bosTokenId ?: 0, eosTokenId ?: 0)
        populateNodes(lattice)
        return lattice.tokens()
    }

    /**
     * Unigram 인코딩을 수행합니다.
     *
     * @param tokens 입력 토큰 리스트 (예: 문장 문자열 리스트)
     * @return 인코딩된 서브토큰 리스트 (각 문자열이 Unigram 모델에 의해 분해되어 리턴됨)
     */
    override fun encode(tokens: List<String>): List<String> {
        val toReturn = mutableListOf<String>()
        for (token in tokens) {
            val tokenized = tokenize(token)
            toReturn.addAll(tokenized)
        }
        return toReturn
    }
}