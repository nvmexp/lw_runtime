<!-- XSLT script to colwert doxygen function xml to html -->
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
  <xsl:template match="/">
    <html>
      <body>
        <!-- Load all doxgen generated xml files -->
        <xsl:for-each select="//memberdef[@kind='function']">
          <b>
            Function 
            <xsl:value-of select="type"/>
            <xsl:text> </xsl:text>
            <xsl:value-of select="name"/>
            <!-- TODO: Format argstring below by replacing it <param siblings> -->
            <xsl:value-of select="argsstring"/>
          </b>
          <xsl:for-each select="briefdescription/para">
            <p style="margin-left:80px"><xsl:value-of select="."/></p>
          </xsl:for-each>

          <xsl:for-each select="detaileddescription/para/text()">
              <p style="margin-left:80px"><xsl:value-of select="."/></p>
          </xsl:for-each>

          <!-- Added for pre-conditions, action, post-conditions sections of LwSciStream public APIs -->
          <xsl:for-each select="detaileddescription/para">
            <xsl:if test="bold">
              <br/><p style="margin-left:80px"><b><xsl:value-of select="bold/."/></b><xsl:value-of select="text()"/></p>
            </xsl:if>

            <xsl:if test="itemizedlist/listitem">
               <xsl:for-each select="itemizedlist/listitem/para">
                 <ul style="margin-left:120px">
                   <li>
                     <xsl:value-of select="."/>
                   </li>
                 </ul>
               </xsl:for-each>
            </xsl:if>
          </xsl:for-each>

          <xsl:if test="detaileddescription/para/simplesect[@kind='note']">
            <br/><p style="margin-left:80px"><b>Notes</b></p>
            <xsl:for-each select="detaileddescription/para/simplesect[@kind='note']/para">
              <p style="margin-left:80px"><xsl:value-of select="."/></p>
            </xsl:for-each>
          </xsl:if>

          <xsl:for-each select="detaileddescription/para">
            <xsl:if test="parameterlist[@kind='templateparam']">
             <br/><p style="margin-left:80px"><b>Template Parameters</b></p>
            </xsl:if>

            <xsl:for-each select="parameterlist[@kind='templateparam']/parameteritem">
              <ul style="margin-left:120px">
                <li>
                  <b><xsl:value-of select="parameternamelist/parametername"/></b>
                  <xsl:text>: </xsl:text>
                  <xsl:choose>
                    <xsl:when test="parameterdescription/para/itemizedlist">
                      <xsl:value-of select="parameterdescription/para/node()[not(self::itemizedlist)]"/>
                    </xsl:when>
                    <xsl:otherwise>
                      <xsl:value-of select="parameterdescription/para/."/>
                    </xsl:otherwise>
                  </xsl:choose>
                  <xsl:for-each select="parameterdescription/para/itemizedlist/listitem">
                    <ul style="margin-left:40px">
                      <li>
                        <xsl:value-of select="."/>
                      </li>
                    </ul>
                  </xsl:for-each>
                </li>
              </ul>
            </xsl:for-each>

            <xsl:if test="parameterlist[@kind='param']">
             <br/><p style="margin-left:80px"><b>Parameters</b></p>
            </xsl:if>

            <xsl:for-each select="parameterlist[@kind='param']/parameteritem">
              <ul style="margin-left:120px">
                <li>
                  [<xsl:value-of select="parameternamelist/parametername/@direction"/>]
                  <xsl:text> </xsl:text>
                  <xsl:value-of select="parameternamelist/parametername"/>
                  <xsl:text>: </xsl:text>
                  <xsl:choose>
                    <xsl:when test="parameterdescription/para/itemizedlist">
                      <xsl:value-of select="parameterdescription/para/node()[not(self::itemizedlist)]"/>
                    </xsl:when>
                    <xsl:otherwise>
                      <xsl:value-of select="parameterdescription/para/."/>
                    </xsl:otherwise>
                  </xsl:choose>
                  <xsl:for-each select="parameterdescription/para/itemizedlist/listitem">
                    <ul style="margin-left:40px">
                      <li>
                        <xsl:value-of select="."/>
                      </li>
                    </ul>
                  </xsl:for-each>
                </li>
              </ul>
            </xsl:for-each>
            <xsl:if test="simplesect[@kind='return']">
              <br/> <p style="margin-left:80px">
                <b>Returns</b><xsl:text> </xsl:text>
                <br/>
                <xsl:value-of select="simplesect[@kind='return']/para/node()[not(self::itemizedlist)]"/>
                <br/>
                <xsl:for-each select="simplesect[@kind='return']/para/itemizedlist/listitem">
                  <ul style="margin-left:120px">
                    <li>
                      <xsl:choose>
                        <xsl:when test="para/itemizedlist">
                          <xsl:value-of select="para/node()[not(self::itemizedlist)]"/>
                        </xsl:when>
                        <xsl:otherwise>
                          <xsl:value-of select="para/."/>
                        </xsl:otherwise>
                      </xsl:choose>
                      <!-- <xsl:value-of select="para/."/> -->
                      <xsl:for-each select="para/itemizedlist/listitem">
                        <ul style="margin-left:140px">
                          <li>
                            <xsl:value-of select="."/>
                          </li>
                        </ul>
                      </xsl:for-each>
                    </li>
                  </ul>
                </xsl:for-each>
              </p>
            </xsl:if>
            <xsl:if test="simplesect[@kind='par']">
              <br/> <p style="margin-left:80px">
                <b>Usage considerations</b><xsl:text> </xsl:text>
                <br/>
                <xsl:value-of select="simplesect[@kind='par']/para/node()[not(self::itemizedlist)]"/>
                <br/>
                <xsl:for-each select="simplesect[@kind='par']/para/itemizedlist/listitem">
                  <ul style="margin-left:120px">
                    <li>
                      <xsl:choose>
                        <xsl:when test="para/itemizedlist">
                          <xsl:value-of select="para/node()[not(self::itemizedlist)]"/>
                        </xsl:when>
                        <xsl:otherwise>
                          <xsl:value-of select="para/."/>
                        </xsl:otherwise>
                      </xsl:choose>
                      <!-- <xsl:value-of select="para/."/> -->
                      <xsl:for-each select="para/itemizedlist/listitem">
                        <ul style="margin-left:140px">
                          <li>
                            <xsl:value-of select="."/>
                          </li>
                        </ul>
                      </xsl:for-each>
                    </li>
                  </ul>
                </xsl:for-each>
              </p>
            </xsl:if>
          </xsl:for-each>
          <br/>
          <h4 style="margin-left:40px;color:#76b900;">File Location</h4>
          <p style="margin-left:80px">
            <xsl:value-of select="location/@file"/>
          </p>
        </xsl:for-each>
      </body>
    </html>
  </xsl:template>
</xsl:stylesheet>
