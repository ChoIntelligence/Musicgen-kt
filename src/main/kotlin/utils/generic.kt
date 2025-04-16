package utils

/**
 * Callable은 함수를 호출하듯이 객체를 호출할 수 있도록 하는 베이스 클래스입니다.
 *
 * 사용 예)
 *
 * open class MyCallable : utils.Callable() {
 *     override fun _call(vararg args: Any?): Any? {
 *         // 인자로 받은 값을 처리하는 로직 구현
 *         return "Called with: ${args.joinToString()}"
 *     }
 * }
 *
 * val myCallable = MyCallable()
 * println(myCallable(1, 2, 3))  // "Called with: 1, 2, 3" 출력
 */
abstract class Callable {
    /**
     * 객체를 함수처럼 호출할 수 있게 합니다.
     *
     * @param args 호출 시 전달되는 인자들
     * @return _call 메서드의 결과
     */
    operator fun invoke(vararg args: Any?): Any? = _call(*args)

    /**
     * 서브클래스에서 반드시 구현해야 하는 메서드입니다.
     *
     * @param args 처리할 인자들
     * @return _call의 처리 결과
     * @throws NotImplementedError 구현되지 않은 경우 예외 발생
     */
    protected abstract fun _call(vararg args: Any?): Any?
}
