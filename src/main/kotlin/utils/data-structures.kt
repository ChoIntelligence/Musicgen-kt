package utils

/**
 * Trie 구조: 문자열을 효율적으로 저장 및 검색하기 위한 자료구조.
 */
class CharTrie {
    var root: CharTrieNode = CharTrieNode.default()

    /**
     * 여러 문자열을 Trie에 추가합니다.
     *
     * @param texts 추가할 문자열 목록
     */
    fun extend(texts: List<String>) {
        texts.forEach { push(it) }
    }

    /**
     * 하나의 문자열을 Trie에 추가합니다.
     *
     * @param text 추가할 문자열
     */
    fun push(text: String) {
        var node = root
        for (ch in text) {
            val child = node.children.getOrPut(ch) { CharTrieNode.default() }
            node = child
        }
        node.isLeaf = true
    }

    /**
     * 주어진 접두사를 가지는 모든 문자열을 검색합니다.
     *
     * @param text 검색할 접두사 문자열
     * @return 접두사를 가진 문자열 목록
     */
    fun commonPrefixSearch(text: String): List<String> {
        var node = root
        val prefixes = mutableListOf<String>()
        var prefix = ""
        for (ch in text) {
            prefix += ch
            node = node.children[ch] ?: break
            if (node.isLeaf) {
                prefixes.add(prefix)
            }
        }
        return prefixes
    }
}

/**
 * Trie의 노드 클래스.
 */
class CharTrieNode(
    var isLeaf: Boolean,
    val children: MutableMap<Char, CharTrieNode>
) {
    companion object {
        fun default(): CharTrieNode = CharTrieNode(false, mutableMapOf())
    }
}


/**
 * TokenLattice는 주어진 문장에 대해 토큰화 후보 노드들을 저장하고,
 * Viterbi 알고리즘을 이용하여 가장 높은 점수를 가진 토큰 시퀀스를 찾기 위한 자료구조입니다.
 *
 * @param sentence 입력 문장
 * @param bosTokenId 시작 토큰 ID
 * @param eosTokenId 끝 토큰 ID
 */
class TokenLattice(sentence: String, val bosTokenId: Int, val eosTokenId: Int) {

    // 문장을 문자 리스트로 저장
    val chars: List<Char> = sentence.toList()
    val len: Int = chars.size

    // 전체 노드 목록
    val nodes: MutableList<TokenLatticeNode> = mutableListOf()

    // 각 위치별 시작 노드 리스트 (길이 len+1 배열)
    val beginNodes: List<MutableList<TokenLatticeNode>> =
        List(len + 1) { mutableListOf() }

    // 각 위치별 종료 노드 리스트 (길이 len+1 배열)
    val endNodes: List<MutableList<TokenLatticeNode>> =
        List(len + 1) { mutableListOf() }

    init {
        // bos: 시작 노드, pos=0, length=0, score=0.0, nodeId = 0
        val bos = TokenLatticeNode(bosTokenId, 0, 0, 0, 0.0)
        // eos: 끝 노드, pos=len, length=0, score=0.0, nodeId = 1
        val eos = TokenLatticeNode(eosTokenId, 1, len, 0, 0.0)
        nodes.add(bos.clone())
        nodes.add(eos.clone())
        beginNodes[len].add(eos)
        endNodes[0].add(bos)
    }

    /**
     * 새로운 토큰 노드를 lattice에 삽입합니다.
     *
     * @param pos 토큰의 시작 위치
     * @param length 토큰의 길이
     * @param score 토큰 점수
     * @param tokenId 토큰 ID
     */
    fun insert(pos: Int, length: Int, score: Double, tokenId: Int) {
        val nodeId = nodes.size
        val node = TokenLatticeNode(tokenId, nodeId, pos, length, score)
        beginNodes[pos].add(node)
        endNodes[pos + length].add(node)
        nodes.add(node)
    }

    /**
     * Viterbi 알고리즘을 이용하여 가장 높은 점수의 토큰 시퀀스를 계산합니다.
     *
     * @return 가장 높은 점수를 가진 토큰 노드 시퀀스 (결과가 없으면 빈 리스트 반환)
     */
    fun viterbi(): List<TokenLatticeNode> {
        var pos = 0
        while (pos <= len) {
            if (beginNodes[pos].isEmpty()) {
                return emptyList()
            }
            for (rnode in beginNodes[pos]) {
                rnode.prev = null
                var bestScore = 0.0
                var bestNode: TokenLatticeNode? = null
                for (lnode in endNodes[pos]) {
                    val score = lnode.backtraceScore + rnode.score
                    if (bestNode == null || score > bestScore) {
                        bestNode = lnode.clone()
                        bestScore = score
                    }
                }
                if (bestNode != null) {
                    rnode.prev = bestNode
                    rnode.backtraceScore = bestScore
                } else {
                    return emptyList()
                }
            }
            pos++
        }
        val results = mutableListOf<TokenLatticeNode>()
        // beginNodes[len]의 첫 번째 노드의 prev가 최종 경로의 마지막 노드가 됨
        val root = beginNodes[len].first()
        val prev = root.prev ?: return emptyList()
        var node = prev.clone()
        while (node.prev != null) {
            results.add(node.clone())
            val n = node.clone()
            node = n.prev!!.clone()
        }
        results.reverse()
        return results
    }

    /**
     * 주어진 노드에 해당하는 토큰(문자열)을 반환합니다.
     *
     * @param node 토큰 노드
     * @return 토큰 문자열
     */
    fun piece(node: TokenLatticeNode): String {
        return chars.subList(node.pos, node.pos + node.length).joinToString("")
    }

    /**
     * Viterbi 알고리즘을 통해 가장 높은 점수를 가진 토큰 시퀀스를 문자열 리스트로 반환합니다.
     *
     * @return 토큰 문자열 리스트
     */
    fun tokens(): List<String> {
        val nodes = viterbi()
        return nodes.map { piece(it) }
    }

    /**
     * Viterbi 알고리즘을 통해 가장 높은 점수를 가진 토큰 시퀀스를 토큰 ID 리스트로 반환합니다.
     *
     * @return 토큰 ID 리스트
     */
    fun tokenIds(): List<Int> {
        val nodes = viterbi()
        return nodes.map { it.tokenId }
    }
}

/**
 * TokenLatticeNode는 토큰 lattice의 각 노드를 나타냅니다.
 *
 * @param tokenId 토큰 ID
 * @param nodeId 노드 ID
 * @param pos 토큰 시작 위치
 * @param length 토큰 길이
 * @param score 토큰 점수
 */
class TokenLatticeNode(
    val tokenId: Int,
    val nodeId: Int,
    val pos: Int,
    val length: Int,
    val score: Double
) {
    var prev: TokenLatticeNode? = null
    var backtraceScore: Double = 0.0

    /**
     * 현재 노드의 복제본을 반환합니다.
     *
     * @return 복제된 utils.TokenLatticeNode
     */
    fun clone(): TokenLatticeNode {
        val copy = TokenLatticeNode(tokenId, nodeId, pos, length, score)
        copy.prev = this.prev
        copy.backtraceScore = this.backtraceScore
        return copy
    }
}
