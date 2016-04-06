

object  MaskHelper {
  val REL_LOVE					            	= 1
  val REL_SPOUSE                      = 2
  val REL_PARENT                      = 3
  val REL_CHILD                       = 4
  val REL_BROTHER_SISTER              = 5
  val REL_UNCLE_AUNT                  = 6
  val REL_RELATIVE                    = 7
  val REL_CLOSE_FRIEND                = 8
  val REL_COLLEAGUE                   = 9
  val REL_SCHOOLMATE                  = 10
  val REL_NEPHEW                      = 11
  val REL_GRANDPARENT                 = 12
  val REL_GRANDCHILD                  = 13
  val REL_COLLEGE_UNIVERSITY_FELLOW   = 14
  val REL_ARMY_FELLOW                 = 15
  val REL_PARENT_IN_LAW               = 16
  val REL_CHILD_IN_LAW                = 17
  val REL_GODPARENT                   = 18
  val REL_GODCHILD                    = 19
  val REL_PLAYING_TOGETHER            = 20

  val GROUP_STRONG_RELATION_MASK = (1 << REL_LOVE) | (1 << REL_SPOUSE) | (1 << REL_PARENT) | (1 << REL_CHILD) |
                              (1 << REL_BROTHER_SISTER) | (1 << REL_UNCLE_AUNT) | (1 << REL_RELATIVE)

  val GROUP_WEAK_RELATION_MASK = (1 << REL_GRANDPARENT) | (1 << REL_GRANDCHILD) | (1 << REL_PARENT_IN_LAW) |
                                  (1 << REL_CHILD_IN_LAW) | (1 << REL_GODPARENT) | (1 << REL_GODCHILD) | (1 << REL_NEPHEW)

  val GROUP_COLLEAGUE_MASK = (1 << REL_COLLEAGUE) | (1 << REL_COLLEGE_UNIVERSITY_FELLOW)

  val GROUP_SCHOOLMATE_MASK = (1 << REL_SCHOOLMATE)

  val GROUP_ARMY_FELLOW_MASK = (1 << REL_ARMY_FELLOW)

  val GROUP_OTHER_MASK = (1 << REL_PLAYING_TOGETHER) | (1 << REL_CLOSE_FRIEND)

  def checkGroup(mask: Int, group: Int) = {
    val maskClear = mask & 0xFFFFFFFE
    if ((maskClear & group) != 0) 1 else 0
  }

  def isStrongRelation(mask: Int) = checkGroup(mask, GROUP_STRONG_RELATION_MASK)

  def isWeakRelation(mask: Int) = checkGroup(mask, GROUP_WEAK_RELATION_MASK)

  def isColleague(mask: Int) = checkGroup(mask, GROUP_COLLEAGUE_MASK)

  def isSchoolmate(mask: Int) = checkGroup(mask, GROUP_SCHOOLMATE_MASK)

  def isArmyFellow(mask: Int) = checkGroup(mask, GROUP_ARMY_FELLOW_MASK)

  def isOther(mask: Int) = checkGroup(mask, GROUP_OTHER_MASK)

}